from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_2bit(x: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values + 1e-8
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_4 = (x_norm * 3).round().to(torch.uint8)  # 2-bit values (0-3)

    packed = torch.zeros(x_quant_4.size(0), x_quant_4.size(1) // 4, dtype=torch.uint8)
    for i in range(4):
        packed |= (x_quant_4[:, i::4] & 0x3) << (2 * i)
    return packed, normalization.to(torch.float16)


def block_dequantize_2bit(x_quant_2: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    device = x_quant_2.device
    unpacked = torch.zeros(x_quant_2.size(0), x_quant_2.size(1) * 4, dtype=torch.float32, device=device)
    for i in range(4):
        unpacked[:, i::4] = ((x_quant_2 >> (2 * i)) & 0x3).to(torch.float32)

    x_norm = unpacked / 3
    normalization = normalization.to(dtype=torch.float32, device=device)
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 128) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        self.register_buffer(
            "weight_q2",
            torch.zeros(out_features * in_features // group_size, group_size // 4, dtype=torch.uint8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.bias = None

        self._register_load_state_dict_pre_hook(Linear2Bit._load_state_dict_pre_hook, with_module=True)

    @staticmethod
    def _load_state_dict_pre_hook(module, state_dict, prefix, *args):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            flat_weight = weight.contiguous().view(-1)
            q2, norm = block_quantize_2bit(flat_weight, group_size=module._group_size)
            module.weight_q2.copy_(q2)
            module.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_2bit(self.weight_q2, self.weight_norm).view(*self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet2Bit(torch.nn.Module):
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


def load(path: Path | None) -> BigNet2Bit:
    net = BigNet2Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
