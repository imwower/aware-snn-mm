from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from snnmm.layers.growth import CoreGrowthManager
from snnmm.layers.neurons import LIFNeuron
from snnmm.layers.oscillation import mean_over_cycles
from snnmm.layers.stdp import rstdp_update_linear


class AwarenessCoreSNN(nn.Module):
    """
    Awareness core with recurrent LIF neurons and classification head.
    Inputs: high-level vision/text representations (time-averaged or spike rates).
    """

    def __init__(
        self,
        vis_dim: int,
        text_dim: int,
        core_dim: int = 192,
        num_classes: int = 100,
        threshold: Optional[float] = None,
        threshold_core: float = 0.5,
        threshold_cls: float = 0.5,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.vis_dim = vis_dim
        self.text_dim = text_dim
        self.core_dim = core_dim
        self.num_classes = num_classes
        if threshold is not None:
            threshold_core = threshold
            threshold_cls = threshold

        self.vis_proj = nn.Linear(vis_dim, core_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, core_dim, bias=False)
        self.recurrent = nn.Linear(core_dim, core_dim, bias=False)
        self.core_neuron = LIFNeuron(threshold=threshold_core, decay=decay)

        self.classifier = nn.Linear(core_dim, num_classes, bias=False)
        self.class_neuron = LIFNeuron(threshold=threshold_cls, decay=decay)
        self.growth_manager: Optional[CoreGrowthManager] = None

        self._init_weights()
        for p in self.parameters():
            p.requires_grad = False

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def forward(
        self,
        h_vis_high: torch.Tensor,
        h_text_sem: torch.Tensor,
        timesteps: int = 20,
        cycle_length: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_vis_high: (B, vis_dim) high-level vision representation.
            h_text_sem: (B, text_dim) text representation.
            timesteps: simulation steps.
            cycle_length: steps per oscillation cycle.
        Returns:
            z_core: time-averaged core spikes (B, core_dim)
            z_vis: time-averaged vis projection in core space (B, core_dim)
            z_text: time-averaged text projection in core space (B, core_dim)
            class_spikes: (T, B, num_classes) spike trains
        """
        B = h_vis_high.shape[0]
        device = h_vis_high.device
        vis_in = self.vis_proj(h_vis_high)
        text_in = self.text_proj(h_text_sem)

        core_state = self.core_neuron.init_state((B, self.core_dim), device=device)
        cls_state = self.class_neuron.init_state((B, self.num_classes), device=device)

        core_spikes = []
        class_spikes = []
        vis_drive = []
        text_drive = []

        with torch.no_grad():
            for _ in range(timesteps):
                # constant input each step from vis/text
                rec_input = self.recurrent(core_state)
                total_in = vis_in + text_in + rec_input
                core_spk, core_state = self.core_neuron(total_in, core_state)
                core_spikes.append(core_spk)
                vis_drive.append(vis_in)
                text_drive.append(text_in)

                cls_in = self.classifier(core_spk)
                cls_spk, cls_state = self.class_neuron(cls_in, cls_state)
                class_spikes.append(cls_spk)

        core_tensor = torch.stack(core_spikes, dim=0)
        vis_tensor = torch.stack(vis_drive, dim=0)
        text_tensor = torch.stack(text_drive, dim=0)
        class_tensor = torch.stack(class_spikes, dim=0)

        z_core = core_tensor.mean(dim=0)
        z_vis = vis_tensor.mean(dim=0)
        z_text = text_tensor.mean(dim=0)

        if self.growth_manager is not None:
            # placeholder surprise zeros; real surprise computed outside
            self.growth_manager.accumulate_sample_stats(z_core, torch.zeros(z_core.shape[0]))

        _ = mean_over_cycles(core_tensor, cycle_length)
        _ = mean_over_cycles(class_tensor, cycle_length)

        return z_core, z_vis, z_text, class_tensor

    @staticmethod
    def compute_surprise(
        class_spikes: torch.Tensor,
        target_labels: torch.Tensor,
        z_vis: torch.Tensor,
        z_text: torch.Tensor,
        alpha: float = 0.7,
        beta: float = 0.3,
    ) -> torch.Tensor:
        """
        Compute surprise S = alpha * classification_confusion + beta * alignment_error.

        Args:
            class_spikes: (T,B,C)
            target_labels: (B,)
            z_vis, z_text: (B, core_dim)
        Returns:
            surprise per sample: (B,)
        """
        rates = class_spikes.mean(dim=0)  # (B,C)
        probs = F.softmax(rates, dim=1)
        batch = target_labels.shape[0]
        correct_probs = probs[torch.arange(batch), target_labels]
        cls_confusion = 1.0 - correct_probs
        align_error = F.mse_loss(z_vis, z_text, reduction="none").mean(dim=1)
        S = alpha * cls_confusion + beta * align_error
        return S

    @staticmethod
    def suggest_gate_delta(surprise: torch.Tensor, num_experts: int = 8, scale: float = 0.1) -> torch.Tensor:
        """
        Produce a simple delta_g suggestion: higher surprise -> flatter gates.
        Returns delta vector (E,) normalized to sum to 0 (mean-zero adjustment).
        """
        base = torch.ones(num_experts) / num_experts
        flatten = base
        delta = flatten - base  # zero vector
        # add small noise scaled by surprise mean to encourage exploration
        delta += scale * surprise.mean().item() * (torch.rand(num_experts) - 0.5)
        return delta

    def attach_growth_manager(self, manager: CoreGrowthManager) -> None:
        self.growth_manager = manager
