from typing import List, Tuple

import torch


def split_cycles(total_steps: int, cycle_length: int) -> List[Tuple[int, int]]:
    """Return list of (start, end) indices for each oscillation cycle."""
    cycles = []
    for start in range(0, total_steps, cycle_length):
        end = min(start + cycle_length, total_steps)
        cycles.append((start, end))
    return cycles


def mean_over_cycles(spike_tensor: torch.Tensor, cycle_length: int) -> torch.Tensor:
    """
    Compute mean spike rate per cycle.

    Args:
        spike_tensor: (T, ...) time-major tensor.
        cycle_length: int, steps per cycle.
    Returns:
        Tensor of shape (num_cycles, ...) with mean over each cycle.
    """
    T = spike_tensor.shape[0]
    cycles = split_cycles(T, cycle_length)
    means = []
    for s, e in cycles:
        means.append(spike_tensor[s:e].mean(dim=0))
    return torch.stack(means, dim=0)
