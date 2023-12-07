import torch
from torch import nn
import torch.nn.functional as F

from Model.NILMFormer.masknet import GLayerNorm
from Model.NILMFormer.TimesNet.TimesNet import TimesBlock
from Model.NILMFormer.TimesNet.Embed import DataEmbedding


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channel

    def forward(self, x: torch.Tensor):
        # B x L -> B x 1 x L
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class TConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation):
        super(TConvBlock, self).__init__()
        self.pwConv1 = nn.Conv1d(in_channel, out_channel, 1, 1)
        self.activation = nn.PReLU()
        self.norm = GLayerNorm(out_channel)
        self.pad = (dilation * (kernel_size - 1)) // 2
        self.dwConv = nn.Conv1d(
            out_channel,
            out_channel,
            kernel_size,
            groups=out_channel,
            padding=self.pad,
            dilation=dilation,
        )
        self.pwConv2 = nn.Conv1d(out_channel, in_channel, 1, 1)

    def forward(self, x):
        out = self.pwConv1(x)
        out = self.activation(out)
        out = self.norm(out)
        out = torch.transpose(out, -1, -2)
        out = self.dwConv(out)
        out = self.activation(out)
        # out = self.norm(out)

        return x + self.pwConv2(out)


class ConvTasNetModule(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super(ConvTasNetModule, self).__init__()
        self.layers = nn.ModuleList(
            [TConvBlock(in_channel, out_channel, 3, 2**i) for i in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class TasEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, depth):
        super(TasEncoder, self).__init__()
        self.norm = GLayerNorm(in_channel)
        self.Conv = nn.Conv1d(in_channel, out_channel, 1, 1)
        self.TasModule = ConvTasNetModule(out_channel, out_channel, depth)

    def forward(self, x):
        x = self.norm(x)
        x = torch.transpose(x, -1, -2)
        x = self.Conv(x)
        x = self.TasModule(x)

        return x


class TimesNetEncoder(nn.Module):
    def __init__(self, in_channel, emb_dim, emb_type, freq, dropout, seq_len, top_k, d_ff, num_kernels):
        super(TimesNetEncoder, self).__init__()
        self.PreConv = Encoder(in_channel, emb_dim, 3)
        self.emb = DataEmbedding(emb_dim, emb_dim, emb_type, freq, dropout)
        self.encoder = TimesBlock(seq_len, top_k, emb_dim, d_ff, num_kernels)

    def forward(self, x, mask=None):
        x = self.PreConv(x)
        x = x.transpose(1, 2)
        x = self.emb(x, mask)
        out = self.encoder(x)
        out = out.transpose(1, 2)

        return out
