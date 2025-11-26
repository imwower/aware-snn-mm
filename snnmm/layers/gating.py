import torch


def update_gates(current: torch.Tensor, delta: torch.Tensor, min_value: float = 0.01) -> torch.Tensor:
    """
    Update gating weights with delta and renormalize.

    Args:
        current: (E,) current gate weights.
        delta: (E,) suggested adjustment.
        min_value: lower bound after normalization.
    Returns:
        new_gates: normalized gate weights summing to 1.
    """
    new = current + delta
    new = torch.clamp(new, min=min_value)
    new = new / (new.sum() + 1e-8)
    return new
