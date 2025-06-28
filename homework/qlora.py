from pathlib import Path
import torch
from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .lower_precision import block_dequantize_2bit, block_quantize_2bit


class QLoRALinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 128,
        bias: bool = True,
    ):
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

        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None

        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False).to(dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False).to(dtype=torch.float32)

        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f32 = x.to(torch.float32)
        weight_f32 = block_dequantize_2bit(self.weight_q2, self.weight_norm).view(self._shape).to(x.device)

        base_out = torch.nn.functional.linear(x_f32, weight_f32, self.bias)
        lora_out = self.lora_b(self.lora_a(x_f32))

        return (base_out + lora_out).to(x.dtype)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight_key = f"{prefix}weight"
        if weight_key in state_dict:
            flat_weight = state_dict[weight_key].contiguous().view(-1)
            q2, norm = block_quantize_2bit(flat_weight, self._group_size)
            self.weight_q2.copy_(q2)
            self.weight_norm.copy_(norm)
            del state_dict[weight_key]
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 128):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
