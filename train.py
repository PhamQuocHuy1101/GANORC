import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import itertools
import os
import time
import argparse
import json
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import Vocab, Spectrogram
from models import Generator, Discriminator
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

torch.backends.cudnn.benchmark = True

def collate_fn(batch):
    images = torch.cat([item[0] for item in batch], dim = 0)
    texts = torch.tensor([item[1] for item in batch], dtype=torch.long)
    length = torch.tensor([item[2] for item in batch])
    # print(images.shape, texts.shape, length.shape)
    return images, texts, length

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    discriminator = Discriminator(h).to(device)

    if rank == 0:
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    cp_g = None
    if os.path.isdir(a.checkpoint_path) and False:
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        print("1")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    print("Load model ...")

    vocab = Vocab(h.vocab)
    trainset = Spectrogram(h.train_file, vocab, h, shuffle=False if h.num_gpus > 1 else True, device=device)
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn = collate_fn)

    if rank == 0:
        validset = Spectrogram(h.val_file, vocab, h, shuffle=False if h.num_gpus > 1 else True, device=device)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True,
                                       collate_fn=collate_fn)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    print("Load dataset")


    generator.train()
    discriminator.train()
    cross_entropy = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            
            spec, r_text, _ = batch
            spec = spec.to(device, non_blocking=True)
            r_text = r_text.to(device, non_blocking=True)

            f_out = generator(spec)
            f_text = torch.argmax(f_out.softmax(dim = -1), dim = -1)
            
            # 0: fake, 1: real
            # Discriminator

            for _ in range(h.gen_interval):
                optim_d.zero_grad()
                d_r_out = discriminator(r_text)
                loss_d_real = cross_entropy(d_r_out, torch.ones(len(d_r_out), device=device, dtype=torch.long))
                # print('--- ', loss_d_real)
                d_f_out = discriminator(f_text.detach())
                loss_d_fake = cross_entropy(d_f_out, torch.zeros(len(d_f_out), device=device, dtype=torch.long))
                # print('+++ ', loss_d_fake)
                loss_disc_all = loss_d_real + loss_d_fake
                loss_disc_all.backward()
                optim_d.step()

            # Generator
            for _ in range(h.gen_interval):
                optim_g.zero_grad()
                
                f_out = generator(spec)
                f_text = torch.argmax(f_out.softmax(dim = -1), dim = -1)
                # MSE loss
                loss_g_mse = mse_loss(f_text.to(dtype=torch.float), r_text.to(dtype=torch.float)) / h.vocab_size
                # cross entropy loss
                d_f_out = discriminator(f_text)
                loss_g_fake = cross_entropy(d_f_out, torch.ones(len(d_f_out), device=device, dtype=torch.long))

                loss_gen_all = loss_g_mse + loss_g_fake
                loss_gen_all.backward()
                optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    print('Step: {:d}, Gen mse {:4.3f}, Gen structure {:4.3f}, Disc real {:4.3f}, Disc fake {:4.3f}'.format(
                        steps, loss_g_mse, loss_g_fake, loss_d_real, loss_d_fake
                    ))

                # checkpointing
                if steps % a.checkpoint_interval == 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'discriminator': (discriminator.module if h.num_gpus > 1
                                                         else discriminator).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/disc_loss_total", loss_disc_all, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        full_seq_acc, char_acc = 0.0, 0.0
                        total_char = 0.0
                        for j, batch in enumerate(tqdm(validation_loader)):
                            spec, r_text, text_length = batch
                            spec = spec.to(device, non_blocking=True)
                            r_text = r_text.to(device, non_blocking=True)
                            
                            f_out = generator(spec)
                            f_text = torch.argmax(f_out.softmax(dim = -1), dim = -1)
                            
                            full_seq_acc += torch.sum(torch.all(f_text == r_text, dim = 1)).item()
                            batch_char_acc = 0.0
                            for l in text_length:
                                batch_char_acc += torch.sum(f_text[:, :l] == r_text[:, :l])
                            char_acc += batch_char_acc
                            total_char += torch.sum(text_length).item()

                        full_seq_acc /= len(validation_loader.dataset)
                        char_acc /= total_char
                        print("Full sequence acc {}, character acc {}".format(full_seq_acc, char_acc))
                        sw.add_scalar("validation/full_seq_acc", full_seq_acc, steps)
                        sw.add_scalar("validation/char_acc", char_acc, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=1000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=500, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
