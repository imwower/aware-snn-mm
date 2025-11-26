from typing import Optional

import torch
from torch import nn

from snnmm.layers.neurons import LIFNeuron


class VisionSNNStage1(nn.Module):
    """Minimal 1-2 layer vision SNN using LIF neurons and STDP."""

    def __init__(
        self,
        input_size: int = 32 * 32 * 3,
        hidden_size: int = 256,
        output_size: int = 128,
        use_second_layer: bool = True,
        threshold: float = 1.0,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.use_second_layer = use_second_layer
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.neuron1 = LIFNeuron(threshold=threshold, decay=decay)
        if use_second_layer:
            self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
            self.neuron2 = LIFNeuron(threshold=threshold, decay=decay)
        self._init_weights()
        for p in self.parameters():
            p.requires_grad = False

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(self, spike_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spike_seq: (T, B, N_in) binary spikes.
        Returns:
            spike record of last layer: (T, B, N_hidden or N_out).
        """
        T, B, _ = spike_seq.shape
        device = spike_seq.device
        h_state = self.neuron1.init_state((B, self.fc1.out_features), device=device)
        out_state: Optional[torch.Tensor] = None
        outputs = []
        with torch.no_grad():
            for t in range(T):
                x_t = spike_seq[t]
                h_input = self.fc1(x_t)
                h_spike, h_state = self.neuron1(h_input, h_state)
                if self.use_second_layer:
                    o_input = self.fc2(h_spike)
                    o_spike, out_state = self.neuron2(o_input, out_state)
                    outputs.append(o_spike)
                else:
                    outputs.append(h_spike)
        return torch.stack(outputs, dim=0)

