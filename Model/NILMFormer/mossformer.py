import torch
from torch import nn

from Model.NILMFormer.encoder import Encoder, TimesNetEncoder
from Model.NILMFormer.decoder import Decoder

from Model.NILMFormer.masknet import MossFormerMaskNet


class MossFormer(nn.Module):
    def __init__(self, config):
        super(MossFormer, self).__init__()

        self.encoder = Encoder(config['in_channel'], config['mid_channel'], config['kernel_size'])
        # self.encoder = TimesNetEncoder(config['in_channel'], config['mid_channel'], 'fixed', 'h', 0.1, 998, 3, config['mid_channel'] * 2, 2)
        self.maskNet = MossFormerMaskNet(config['mid_channel'], config['mid_channel'], config['device_num'], config['depth'])
        self.decoder = Decoder(config['mid_channel'], config['out_channel'], config['kernel_size'])

        self.nums = config['device_num']

    def forward(self, x):
        mix_x = self.encoder(x)
        mask = self.maskNet(mix_x)
        mix_x = torch.stack([mix_x] * self.nums)
        sep = mix_x * mask
        estTarget = torch.cat(
            [self.decoder(sep[i]).unsqueeze(0) for i in range(self.nums)],
            dim=0,
        )

        return estTarget