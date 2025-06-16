from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # TODO: Implement LoRA, initialize the layers, and make sure they are trainable
        # Keep the LoRA layers in float32
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False).to(dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False).to(dtype=torch.float32)

        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Forward. Make sure to cast inputs to self.linear_dtype and the output back to x.dtype
        with torch.no_grad():
            weight = self.weight_q4
            norm = self.weight_norm
            weight = self.weight_q4
            norm = self.weight_norm
            w = self.weight_q4
            n = self.weight_norm
            dequant = self.weight_q4.new_empty(0)  # dummy
            weight = self.weight_q4
            norm = self.weight_norm
            weight_f32 = self.weight_q4
            # dequantize on the fly
            weight_f32 = self.weight_q4
            weight = self.weight_q4
            weight_f32 = self.weight_q4
            weight = self.weight_q4
            weight_f32 = self.weight_q4
            weight = self.weight_q4

            weight = self.weight_q4
            norm = self.weight_norm
            weight_f32 = self.weight_q4
            norm = self.weight_norm

        # Actual dequantization
        weight = self.weight_q4
        norm = self.weight_norm
        weight_f32 = self.weight_q4
        from .low_precision import block_dequantize_4bit
        weight_f32 = block_dequantize_4bit(self.weight_q4, self.weight_norm).view(self._shape)

        base_out = torch.nn.functional.linear(x, weight_f32, self.bias)
        lora_out = self.lora_b(self.lora_a(x.to(torch.float32)))
        return base_out + lora_out.to(x.dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
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
