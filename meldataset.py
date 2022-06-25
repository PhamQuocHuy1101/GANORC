import math
import os
import random
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class Vocab():
    def __init__(self, vocab):
        self.char2token = {'<pad>': 0}
        self.token2char = {0: '<pad>'}
        self.padding_index = 0 # not change
        self.padding_token = '<pad>'

        vocab = list(vocab)
        for i, c in enumerate(vocab):
            self.char2token[c] = i+1
            self.token2char[i+1] = c
        
    def decode(self, ids):
        ''' ids : list of index '''
        seq = [self.token2char[id] for id in ids]
        return seq
    def encode(self, seq):
        ''' seq : list of token '''
        ids = [self.char2token[s] for s in seq]
        return ids

class TransformText:
    def __init__(self, vocab, label_len):
        self.label_len = label_len
        self.vocab = vocab

    def __call__(self, seq):
        ''' seq: sequence '''
        seq, length = self.pad_token(seq)
        return self.vocab.encode(seq), length
    
    def pad_token(self, seq):
        n_padding = self.label_len - len(seq)
        assert n_padding >= 0, "Error:{}".format(seq)

        pad_seq = list(seq) + [self.vocab.padding_token] * n_padding
        return pad_seq, len(seq) + 1

class TransformImage:
    def __init__(self, height, min_width, max_width):
        self.height = height
        self.min_width = min_width
        self.max_width = max_width
    
    def get_size(self, w, h):
        new_w = int(self.height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w/round_to)*round_to
        new_w = max(new_w, self.min_width)
        new_w = min(new_w, self.max_width)
        return new_w

    def transform(self, image):
        '''
            image: numpy 
        '''
        img = F.to_tensor(image)
        h, w = img.shape[1:]

        width_size = self.get_size(w, h)
        img = F.resize(img, (self.height, width_size))#, antialias = True)

        return img
    
    def __call__(self, img):
        output = self.transform(img)
        w = output.shape[-1]
        output = F.pad(output, [0, 0, self.max_width - w, 0], fill = 0, padding_mode = 'constant')
        return output

class Spectrogram(torch.utils.data.Dataset):
    def __init__(self, file, vocab, h, shuffle=True, device=None):

        self.vocab = vocab
        self.records = self.read_annotation(file)
        self.img_dir = h.img_dir
        self.transf_img = TransformImage(h.height, h.min_width, h.max_width)
        self.transf_label = TransformText(vocab, h.max_label_length)
        self.n_fft = h.n_fft
        self.hop_size = h.hop_size
        self.win_size = h.win_size
        self.device = device
        if shuffle:
            random.shuffle(self.records)
        self.hann_window = torch.hann_window(self.win_size)#, device = device)

    def read_annotation(self, file):
        '''
            return: [[img_file, text]]
        '''
        with open(file, 'r') as f:
            data = f.read().splitlines()
            files = [d.split('\t') for d in data]
            return files

    def __getitem__(self, index):
        img_f, text = self.records[index]
        image = Image.open(os.path.join(self.img_dir, img_f)).convert('L')
        image = self.transf_img(image)
        image = image.permute(0, 2, 1).flatten(1)

        spec = torch.stft(image, self.n_fft, 
                        hop_length=self.hop_size, 
                        win_length=self.win_size, 
                        window=self.hann_window,
                        center=False, pad_mode=False, normalized=True, onesided=False, return_complex=False)

        spec = torch.sum(spec.pow(2), dim = -1)
        spec = spec.permute(0, 2, 1)
        text_ids, length = self.transf_label(text)
        return spec, text_ids, length

    def __len__(self):
        return len(self.records)