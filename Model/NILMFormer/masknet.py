import torch
from torch import nn, einsum
import torch.nn.functional as F


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


class Transpose(nn.Module):
    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(*self.shape)


class GLayerNorm(nn.Module):
    def __init__(self, dim=64, eps=1e-5):
        super(GLayerNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        mean = torch.mean(x, dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True)
        x = self.gamma * (x - mean) / (self.eps + std) + self.beta

        return torch.transpose(x, 1, 2)


class DepthwiseConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ) -> None:
        super(DepthwiseConv1d, self).__init__()
        assert (
            out_channels % in_channels == 0
        ), 'out_channels should be constant multiple of in_channels'
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class ScaledSinuEmbedding(nn.Module):
    def __init__(self, dim):
        super(ScaledSinuEmbedding, self).__init__()
        self.dim = dim
        self.scale = nn.Parameter(
            torch.ones(
                1,
            )
        )
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freq', self.inv_freq)

    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)
        self.inv_freq = self.inv_freq.to(device=x.device)
        multi = einsum('i, j -> ij', pos, self.inv_freq)
        pe_sin = multi.sin()
        pe_cos = multi.cos()
        emb = torch.zeros_like(x)
        emb[:, :, 0::2] = pe_sin
        emb[:, :, 1::2] = pe_cos

        return emb * self.scale


class MossFormerConvModule(nn.Module):
    def __init__(
        self, in_channels: int, kernel_size: int = 17, expansion_factor: int = 2
    ) -> None:
        super(MossFormerConvModule, self).__init__()
        assert (
            kernel_size - 1
        ) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, 'Currently, Only Supports expansion_factor 2'

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2)),
            DepthwiseConv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.sequential(inputs).transpose(1, 2)


class ConvM(nn.Module):
    def __init__(self, dim_in, dim_out, norm=nn.LayerNorm, dropout=0.1):
        super().__init__()
        self.mdl = nn.Sequential(
            norm(dim_in),
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            MossFormerConvModule(dim_out),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        output = self.mdl(x)
        return output


class OffsetScale(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))
        self.beta = nn.Parameter(torch.zeros(heads, dim))
        nn.init.normal_(self.gamma, std=0.02)

    def forward(self, x):
        out = einsum('... d, h d -> ... h d', x, self.gamma) + self.beta
        return out.unbind(dim=-2)


class MossFormerBlock(nn.Module):
    def __init__(
        self,
        dim,
        group_size=50,
        query_key_dim=128,
        expansion_factor=2.0,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=False,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens
        # positional embeddings
        self.rotary_pos_emb = rotary_pos_emb
        # norm
        self.dropout = nn.Dropout(dropout)
        # projections
        self.to_hidden = ConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm=norm_klass,
            dropout=dropout,
        )
        self.to_qk = ConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm=norm_klass,
            dropout=dropout,
        )
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)
        self.to_out = ConvM(
            dim_in=dim,
            dim_out=dim,
            dropout=dropout,
        )
        self.gateActivate = nn.Sigmoid()

    def forward(self, x):
        # prenorm
        normed_x = x
        # do token shift - a great, costless trick from an independent AI researcher in Shenzhen
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.0)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # initial projections
        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)
        # offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u)

        # projection out and residual
        out = (att_u * v) * self.gateActivate(att_v * u)
        x = x + self.to_out(out)
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        b, n, device, g = x.shape[0], x.shape[-2], x.device, self.group_size

        from einops import rearrange

        if exists(mask):
            lin_mask = rearrange(mask, '... -> ... 1')
            lin_k = lin_k.masked_fill(~lin_mask, 0.0)

        # rotate queries and keys
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(
                self.rotary_pos_emb.rotate_queries_or_keys,
                (quad_q, lin_q, quad_k, lin_k),
            )

        # padding for groups
        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            quad_q, quad_k, lin_q, lin_k, v, u = map(
                lambda t: F.pad(t, (0, 0, 0, padding), value=0.0),
                (quad_q, quad_k, lin_q, lin_k, v, u),
            )
            mask = default(mask, torch.ones((b, n), device=device, dtype=torch.bool))
            mask = F.pad(mask, (0, padding), value=False)

        # group along sequence
        quad_q, quad_k, lin_q, lin_k, v, u = map(
            lambda t: rearrange(t, 'b (g n) d -> b g n d', n=self.group_size),
            (quad_q, quad_k, lin_q, lin_k, v, u),
        )

        if exists(mask):
            mask = rearrange(mask, 'b (g j) -> b g 1 j', j=g)

        # calculate quadratic attention output
        sim = einsum('... i d, ... j d -> ... i j', quad_q, quad_k) / g
        attn = F.relu(sim) ** 2
        attn = self.dropout(attn)
        if exists(mask):
            attn = attn.masked_fill(~mask, 0.0)
        if self.causal:
            causal_mask = torch.ones((g, g), dtype=torch.bool, device=device).triu(1)
            attn = attn.masked_fill(causal_mask, 0.0)

        quad_out_v = einsum('... i j, ... j d -> ... i d', attn, v)
        quad_out_u = einsum('... i j, ... j d -> ... i d', attn, u)

        # calculate linear attention output
        if self.causal:
            lin_kv = einsum('b g n d, b g n e -> b g d e', lin_k, v) / g
            # exclusive cumulative sum along group dimension
            lin_kv = lin_kv.cumsum(dim=1)
            lin_kv = F.pad(lin_kv, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_v = einsum('b g d e, b g n d -> b g n e', lin_kv, lin_q)

            lin_ku = einsum('b g n d, b g n e -> b g d e', lin_k, u) / g
            # exclusive cumulative sum along group dimension
            lin_ku = lin_ku.cumsum(dim=1)
            lin_ku = F.pad(lin_ku, (0, 0, 0, 0, 1, -1), value=0.0)
            lin_out_u = einsum('b g d e, b g n d -> b g n e', lin_ku, lin_q)
        else:
            lin_kv = einsum('b g n d, b g n e -> b d e', lin_k, v) / n
            lin_out_v = einsum('b g n d, b d e -> b g n e', lin_q, lin_kv)

            lin_ku = einsum('b g n d, b g n e -> b d e', lin_k, u) / n
            lin_out_u = einsum('b g n d, b d e -> b g n e', lin_q, lin_ku)

        # fold back groups into full sequence, and excise out padding
        quad_attn_out_v, lin_attn_out_v = map(
            lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n],
            (quad_out_v, lin_out_v),
        )
        # , lin_attn_out_v
        #  lin_out_v
        quad_attn_out_u, lin_attn_out_u = map(
            lambda t: rearrange(t, 'b g n d -> b (g n) d')[:, :n],
            (quad_out_u, lin_out_u),
        )
        # , lin_attn_out_u
        #  lin_out_u

        # gate
        return quad_attn_out_v + lin_attn_out_v, quad_attn_out_u + lin_attn_out_u


class MossFormerModule(nn.Module):
    def __init__(
        self,
        dim,
        depth=8,
        group_size=50,
        query_key_dim=128,
        expansion_factor=2.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True,
    ):
        super().__init__()
        assert norm_type in (
            'scalenorm',
            'layernorm',
        ), 'norm_type must be one of scalenorm or layernorm'

        norm_klass = nn.LayerNorm

        from rotary_embedding_torch import RotaryEmbedding

        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))
        # max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        self.layers = nn.ModuleList(
            [
                MossFormerBlock(
                    dim=dim,
                    group_size=group_size,
                    query_key_dim=query_key_dim,
                    expansion_factor=expansion_factor,
                    causal=causal,
                    dropout=attn_dropout,
                    rotary_pos_emb=rotary_pos_emb,
                    norm_klass=norm_klass,
                    shift_tokens=False,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for mossformer_layer in self.layers:
            x = mossformer_layer(x)
        return x


class MossFormerMaskNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        nums_devices,
        depth,
        skip_connection=True,
        use_global_pos_enc=True,
    ):
        super(MossFormerMaskNet, self).__init__()
        self.nums_devices = nums_devices
        self.depth = depth
        self.norm = GLayerNorm(out_channels)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.pw_conv = nn.ModuleList(
            [
                nn.Conv1d(in_channels, in_channels, 1, bias=False),
                nn.Conv1d(in_channels, in_channels * nums_devices, 1, bias=False),
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
            ]
        )
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        # self.MFB = MossFormerModule(in_channels, depth)
        self.MFB = MossFormerBlock(dim=in_channels)
        self.prelu = nn.PReLU()
        self.glu = nn.GLU(dim=1)
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )
        self.gn = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.skip_connection = skip_connection

    def forward(self, x: torch.Tensor):
        # _, _, s = x.shape
        x = self.norm(x)
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(-2, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(-2, -1)
            x = base + emb
        x = self.pw_conv[0](x)
        att = x.transpose(-2, -1)
        att = self.MFB(att)
        att = att.transpose(-2, -1)
        x = self.prelu(att)
        x = self.pw_conv[1](x)
        b, _, s = x.shape
        x = x.view(b * self.nums_devices, -1, s)
        x = self.output(x) * self.output_gate(x)
        x = self.pw_conv[2](x)
        x = self.prelu(x)
        _, n, L = x.shape
        x = x.view(b, self.nums_devices, n, L)
        # x = self.activation(x)
        x = x.transpose(0, 1)
        return x


class MossFormerMaskNet_v2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        nums_devices,
        depth,
        skip_connection=True,
        use_global_pos_enc=True,
    ):
        super(MossFormerMaskNet_v2, self).__init__()
        self.nums_devices = nums_devices
        self.depth = depth
        self.norm = GLayerNorm(out_channels)
        self.pw_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.gate = nn.ModuleList([
            GateNet(in_channels, out_channels)
            for _ in range(nums_devices)
        ])
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.MFB = MossFormerBlock(dim=in_channels)
        self.relu = nn.PReLU()
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        if self.use_global_pos_enc:
            base = x
            x = x.transpose(-2, -1)
            emb = self.pos_enc(x)
            emb = emb.transpose(-2, -1)
            x = base + emb
        x = self.norm(x)
        x = self.pw_conv(x)
        att = x.transpose(-2, -1)
        att = self.MFB(att)
        att = att.transpose(-2, -1)
        att = self.relu(att)
        output = []
        for layer in self.gate:
            output.append(layer(att))

        return output


class GateNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GateNet, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channel, out_channel, 1),
            nn.Conv1d(in_channel, out_channel, 1)
        ])
        self.output = nn.Sequential(nn.Conv1d(in_channel, out_channel, 1), nn.Tanh())
        self.output_gate = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1), nn.Sigmoid()
        )
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv[0](x)
        x = self.output(x) * self.output_gate(x)
        x = self.conv[1](x)

        return self.activation(x)