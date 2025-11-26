from typing import Optional

import torch


def label_poisson_encode(
    labels: torch.Tensor,
    timesteps: int = 20,
    n_labels: int = 100,
    high_rate: float = 0.9,
    low_rate: float = 0.01,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    One-hot style Poisson spike encoding for labels.

    Args:
        labels: (B,) int tensor of label indices.
        timesteps: simulation steps.
        n_labels: number of label neurons.
        high_rate: firing prob for correct label per timestep.
        low_rate: firing prob for other labels.
    Returns:
        spikes: (T, B, n_labels) binary tensor.
    """
    labels = labels.to(device) if device is not None else labels
    B = labels.shape[0]
    spikes = torch.zeros((timesteps, B, n_labels), device=labels.device)
    for t in range(timesteps):
        base = torch.full((B, n_labels), low_rate, device=labels.device)
        base[torch.arange(B), labels] = high_rate
        rnd = torch.rand((B, n_labels), device=labels.device)
        spikes[t] = (rnd < base).float()
    return spikes

