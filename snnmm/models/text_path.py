from typing import Optional, Tuple

import torch
from torch import nn

from snnmm.layers.neurons import LIFNeuron


class LabelSNN(nn.Module):
    """Simple multi-layer label/text SNN using LIF neurons and linear layers."""

    def __init__(
        self,
        n_labels: int = 100,
        hidden_size: int = 128,
        semantic_size: int = 128,
        use_third: bool = False,
        threshold: float = 1.0,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.use_third = use_third
        self.fc1 = nn.Linear(n_labels, hidden_size, bias=False)
        self.neuron1 = LIFNeuron(threshold=threshold, decay=decay)
        self.fc2 = nn.Linear(hidden_size, semantic_size, bias=False)
        self.neuron2 = LIFNeuron(threshold=threshold, decay=decay)
        if use_third:
            self.fc3 = nn.Linear(semantic_size, semantic_size, bias=False)
            self.neuron3 = LIFNeuron(threshold=threshold, decay=decay)
        self._init_weights()
        for p in self.parameters():
            p.requires_grad = False

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(self, spike_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            spike_seq: (T, B, n_labels) binary spikes.
        Returns:
            h_text_low: time-averaged first hidden spikes.
            h_text_sem: time-averaged semantic spikes (final layer).
        """
        T, B, _ = spike_seq.shape
        device = spike_seq.device
        s1 = self.neuron1.init_state((B, self.fc1.out_features), device=device)
        s2 = self.neuron2.init_state((B, self.fc2.out_features), device=device)
        s3: Optional[torch.Tensor] = None
        if self.use_third:
            s3 = self.neuron3.init_state((B, self.fc3.out_features), device=device)

        spk1_all = []
        spk2_all = []
        spk3_all = []
        with torch.no_grad():
            for t in range(T):
                x_t = spike_seq[t]
                h1_in = self.fc1(x_t)
                spk1, s1 = self.neuron1(h1_in, s1)
                h2_in = self.fc2(spk1)
                spk2, s2 = self.neuron2(h2_in, s2)
                spk1_all.append(spk1)
                spk2_all.append(spk2)
                if self.use_third:
                    h3_in = self.fc3(spk2)
                    spk3, s3 = self.neuron3(h3_in, s3)
                    spk3_all.append(spk3)

        spk1_tensor = torch.stack(spk1_all, dim=0)
        spk2_tensor = torch.stack(spk2_all, dim=0)
        if self.use_third:
            spk3_tensor = torch.stack(spk3_all, dim=0)
            return spk1_tensor.mean(dim=0), spk3_tensor.mean(dim=0)
        return spk1_tensor.mean(dim=0), spk2_tensor.mean(dim=0)

