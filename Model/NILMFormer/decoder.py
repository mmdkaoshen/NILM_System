import torch
from torch import nn


class Decoder(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise RuntimeError('{} accept 3/4D tensor as input'.format(self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class UDecoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1) -> None:
        super(UDecoder, self).__init__()
        self.conv_first = nn.Conv1d(
            in_channel, in_channel, kernel_size, padding=kernel_size // 2
        )
        self.conv_mid = nn.Conv1d(
            in_channel,
            in_channel,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.conv_last = nn.Conv1d(
            in_channel, out_channel, kernel_size, padding=kernel_size // 2
        )
        self.conv_block = VConvBlock(
            in_channel=in_channel,
            out_channel=in_channel * 2,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.conv_transpose = nn.ConvTranspose1d(
            in_channel * 2, in_channel, kernel_size, 2, kernel_size // 2, 1
        )
        self.max_pool = nn.MaxPool1d(1, 2)

    def forward(self, x):
        x = self.conv_first(x)
        down1 = self.conv_block(x)
        down1 = self.max_pool(down1)

        down2 = self.conv_block(down1)
        down2 = self.max_pool(down2)

        down3 = self.conv_block(down2)
        down3 = self.max_pool(down3)

        mid = self.conv_mid(down3)

        up3 = self.conv_block(mid)
        up3 = torch.cat([up3, down3], dim=1)
        up3 = self.conv_transpose(up3)

        up2 = self.conv_block(up3)
        up2 = torch.cat([up3, down2], dim=1)
        up2 = self.conv_transpose(up2)

        up1 = self.conv_block(up2)
        up1 = torch.cat([up1, down1], dim=1)
        up1 = self.conv_transpose(up1)

        output = self.conv_last(up1)

        if torch.squeeze(output).dim() == 1:
            output = torch.squeeze(output, dim=1)
        else:
            output = torch.squeeze(output)

        return output


class VConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(VConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channel, out_channel, kernel_size, stride, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(out_channel),
            nn.PReLU(),
            nn.Conv1d(out_channel, out_channel, 1, stride),
            nn.BatchNorm1d(out_channel),
            nn.PReLU(),
            nn.Conv1d(
                out_channel, in_channel, kernel_size, stride, padding=kernel_size // 2
            ),
            nn.BatchNorm1d(in_channel),
        )
        self.instance_norm = nn.InstanceNorm1d(in_channel)
        self.activation = nn.PReLU()

    def forward(self, x):
        x_res = self.conv(x)
        x = x + x_res
        x = self.instance_norm(x)
        x = self.activation(x)

        return x
