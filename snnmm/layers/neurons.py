from typing import Optional, Tuple

import torch
from torch import nn


class LIFNeuron(nn.Module):
    """Leaky integrate-and-fire neuron with simple reset."""

    def __init__(self, threshold: float = 1.0, decay: float = 0.95, reset_value: float = 0.0):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.reset_value = reset_value

    def init_state(self, shape: Tuple[int, ...], device: Optional[torch.device] = None) -> torch.Tensor:
        return torch.zeros(shape, device=device)

    def forward(
        self,
        input_spikes: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # No autograd; states are updated explicitly.
        with torch.no_grad():
            v = state if state is not None else torch.zeros_like(input_spikes)
            v = v * self.decay + input_spikes
            spikes = (v >= self.threshold).float()
            reset_mask = spikes > 0
            v = torch.where(reset_mask, torch.full_like(v, self.reset_value), v)
        return spikes, v

