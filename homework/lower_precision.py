from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_2bit(x: torch.Tensor, group_size: int = 16):
    assert x.dim() == 1 and x.numel() % group_size == 0
    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values + 1e-8
    x_norm = (x + normalization) / (2 * normalization)
    x_quant = (x_norm * 3).round().clamp(0, 3).to(torch.uint8)
    packed = (x_quant[:, ::4] |
              (x_quant[:, 1::4] << 2) |
              (x_quant[:, 2::4] << 4) |
              (x_quant[:, 3::4] << 6))
    return packed, normalization.to(torch.float16)


def block_dequantize_2bit(packed: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    unpacked = torch.empty(packed.shape[0], packed.shape[1] * 4, dtype=torch.float32, device=packed.device)
    for i in range(4):
        unpacked[:, i::4] = ((packed >> (2 * i)) & 0x3).float()
    x_norm = unpacked / 3
    normalization = normalization.to(torch.float32)
    return (x_norm * 2 * normalization) - normalization


class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16):
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        num_groups = (in_features * out_features) // group_size
        self.register_buffer("weight_q2", torch.zeros(num_groups, group_size // 4, dtype=torch.uint8), persistent=False)
        self.register_buffer("weight_norm", torch.zeros(num_groups, 1, dtype=torch.float16), persistent=False)

        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else None
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(self, state_dict, prefix, *_):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            flat_weight = weight.view(-1)
            q2, norm = block_quantize_2bit(flat_weight, self._group_size)
            self.weight_q2.copy_(q2)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = block_dequantize_2bit(self.weight_q2, self.weight_norm).view(*self._shape)
        return torch.nn.functional.linear(x.to(torch.float32), weight, self.bias)


class LowerPrecisionBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LowerPrecisionBigNet:
    net = LowerPrecisionBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
