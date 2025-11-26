from typing import Tuple

import torch


def poisson_encode(
    images: torch.Tensor,
    timesteps: int = 20,
    max_rate: float = 1.0,
    flatten: bool = True,
) -> torch.Tensor:
    """
    Rate-based Poisson encoding for RGB images.

    Args:
        images: float tensor of shape (B, 3, 32, 32) in [0, 1].
        timesteps: number of simulation steps.
        max_rate: maximum firing probability per step.
        flatten: if True, flatten spatial dims to (B, N_in); else keep (B, 3, 32, 32).

    Returns:
        spikes: tensor of shape (T, B, N_in) with binary values (0/1).
    """
    if images.max() > 1.0:
        images = images / 255.0
    images = images.clamp(0.0, 1.0)
    batch_shape: Tuple[int, ...]
    if flatten:
        batch_shape = (images.shape[0], images.numel() // images.shape[0])
        rates = images.view(batch_shape)
    else:
        batch_shape = images.shape
        rates = images

    # Probability per timestep scaled by max_rate.
    probs = rates * max_rate
    spikes = torch.zeros((timesteps,) + batch_shape, device=images.device, dtype=torch.float32)
    for t in range(timesteps):
        rand = torch.rand(batch_shape, device=images.device)
        spikes[t] = (rand < probs).float()
    return spikes

