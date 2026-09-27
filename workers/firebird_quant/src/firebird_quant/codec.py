"""Real packed signed 4/8-bit weights; no model names or operator assumptions."""

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor

DTYPES = {
    str(dtype): dtype for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
}


@dataclass(frozen=True)
class PackedTensor:
    codes: Tensor
    scales: Tensor
    shape: tuple[int, ...]
    dtype: str
    bits: int
    group_size: int

    @property
    def nbytes(self):
        return self.codes.numel() + self.scales.numel() * self.scales.element_size()

    def validate(self):
        if type(self.bits) is not int or self.bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        if type(self.group_size) is not int or self.group_size < 2 or self.group_size % 2:
            raise ValueError("group_size must be a positive even integer >= 2")
        if self.dtype not in DTYPES or any(type(n) is not int or n < 0 for n in self.shape):
            raise ValueError("Invalid original tensor shape or dtype")
        groups = (prod(self.shape) + self.group_size - 1) // self.group_size
        scale_dtype = torch.float64 if self.dtype == "torch.float64" else torch.float32
        if (
            self.codes.dtype != torch.uint8
            or tuple(self.codes.shape) != (groups * self.group_size * self.bits // 8,)
            or self.scales.dtype != scale_dtype
            or tuple(self.scales.shape) != (groups,)
            or not torch.isfinite(self.scales).all()
            or not (self.scales > 0).all()
            or not torch.isfinite(
                (self.scales * ((1 << (self.bits - 1)) - 1)).to(DTYPES[self.dtype])
            ).all()
        ):
            raise ValueError("Invalid packed tensor storage or scales")

    def dequantize(self):
        return unpack(
            self.codes, self.scales, self.shape, DTYPES[self.dtype], self.bits, self.group_size
        )


def unpack(codes, scales, shape, dtype, bits, group_size):
    """Ordinary Torch operations run on the buffers' device, without special kernels."""
    if bits == 4:
        values = torch.stack((codes & 15, codes >> 4), dim=-1).flatten().to(scales.dtype) - 8
    else:
        values = codes.view(torch.int8).to(scales.dtype)
    values = values.reshape(-1, group_size) * scales[:, None]
    return values.flatten()[: prod(shape)].reshape(shape).to(dtype)


def pack(tensor: Tensor, *, bits=4, group_size=64) -> PackedTensor:
    if type(bits) is not int or bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    if type(group_size) is not int or group_size < 2 or group_size % 2:
        raise ValueError("group_size must be a positive even integer >= 2")
    if str(tensor.dtype) not in DTYPES or tensor.layout != torch.strided or tensor.is_meta:
        raise ValueError("Packing requires a materialized, dense real floating tensor")
    dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    values = tensor.detach().to(device="cpu", dtype=dtype).contiguous().flatten()
    if not torch.isfinite(values).all():
        raise ValueError("Cannot quantize NaN or infinite weights")
    groups = (values.numel() + group_size - 1) // group_size
    padded = torch.zeros(groups * group_size, dtype=dtype)
    padded[: values.numel()] = values
    blocks = padded.reshape(groups, group_size)
    qmax = (1 << (bits - 1)) - 1
    maxima = blocks.abs().amax(dim=1)
    # Clamp subnormal scales to avoid division by zero. Zero blocks use scale 1.
    scales = (maxima / qmax).clamp_min(torch.finfo(dtype).tiny)
    scales = torch.where(maxima == 0, torch.ones_like(scales), scales)
    # At the dtype maximum, division can round upward and multiplication overflow.
    scales = torch.where(
        torch.isfinite((scales * qmax).to(tensor.dtype)),
        scales,
        torch.nextafter(scales, torch.zeros_like(scales)),
    )
    quantized = (blocks / scales[:, None]).round().clamp(-qmax, qmax).flatten()
    if bits == 4:
        unsigned = (quantized + 8).to(torch.uint8)
        codes = unsigned[::2] | (unsigned[1::2] << 4)
    else:
        codes = quantized.to(torch.int8).view(torch.uint8)
    result = PackedTensor(codes, scales, tuple(tensor.shape), str(tensor.dtype), bits, group_size)
    result.validate()
    return result
