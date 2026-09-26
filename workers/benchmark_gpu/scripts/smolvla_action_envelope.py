"""Lossless FP32 transport for a single preprocessed SmolVLA observation."""

import torch

ENVELOPE_WIDTH = 2048
IMAGE_VALUES = 3 * 512 * 512
PAYLOAD_VALUES = 2 * IMAGE_VALUES + 32 + 48 + 48 + 50 * 32
ENVELOPE_TOKENS = (PAYLOAD_VALUES + ENVELOPE_WIDTH - 1) // ENVELOPE_WIDTH


def unpack(payload):
    if payload.shape != (ENVELOPE_TOKENS, ENVELOPE_WIDTH) or payload.dtype != torch.float32:
        raise ValueError("One complete FP32 SmolVLA observation envelope is required")
    flat = payload.reshape(-1)
    offset = 0

    def take(n):
        nonlocal offset
        value = flat[offset : offset + n]
        offset += n
        return value

    images = [take(IMAGE_VALUES).reshape(1, 3, 512, 512) for _ in range(2)]
    state = take(32).reshape(1, 32)
    tokens = take(48).reshape(1, 48).long()
    masks = take(48).reshape(1, 48).bool()
    noise = take(1600).reshape(1, 50, 32)
    image_masks = [torch.ones(1, device=flat.device, dtype=torch.bool) for _ in range(2)]
    return images, image_masks, tokens, masks, state, noise
