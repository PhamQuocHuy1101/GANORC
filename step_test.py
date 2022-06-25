import os
import json
import torch

from env import AttrDict
from models import Generator, Discriminator, MultiScaleDiscriminator, feature_loss, generator_loss

def test_stft():
    n_fft = 256
    hop_size = 8
    win_size= 32
    window = torch.hann_window(win_size)


    y = torch.randn(2, 32 * 512)


    out = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=window,
                        center=False, pad_mode='reflect', normalized=False, onesided=False)
    print(out.shape, window.shape)

if __name__ == '__main__':
    config = 'config_v1.json'
    with open(config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h)
    discriminator = Discriminator(h)

    sample = torch.randn(2, 2017, 256)
    out_g = generator(sample)
    out_text = torch.argmax(out_g.softmax(dim = -1), dim = -1)
    out_d = discriminator(out_text)
    print("G: ", out_g.shape)
    print("D: ", out_d.shape)
    

