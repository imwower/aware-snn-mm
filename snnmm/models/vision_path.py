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


class VisionMultiStageSNN(nn.Module):
    """
    Three-stage vision SNN with 8 experts per stage and gating weights.
    Inputs are flattened to (B, N) if needed.
    Returns time-averaged outputs for each stage for downstream alignment.
    """

    def __init__(
        self,
        input_size: int = 32 * 32 * 3,
        stage1_size: int = 256,
        stage2_size: int = 256,
        stage3_size: int = 128,
        num_experts: int = 8,
        threshold: float = 1.0,
        decay: float = 0.95,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.gates = nn.ParameterDict(
            {
                "stage1": nn.Parameter(torch.full((num_experts,), 1.0 / num_experts), requires_grad=False),
                "stage2": nn.Parameter(torch.full((num_experts,), 1.0 / num_experts), requires_grad=False),
                "stage3": nn.Parameter(torch.full((num_experts,), 1.0 / num_experts), requires_grad=False),
            }
        )
        self.stage1 = nn.ModuleList(
            [nn.Linear(input_size, stage1_size, bias=False) for _ in range(num_experts)]
        )
        self.stage2 = nn.ModuleList(
            [nn.Linear(stage1_size, stage2_size, bias=False) for _ in range(num_experts)]
        )
        self.stage3 = nn.ModuleList(
            [nn.Linear(stage2_size, stage3_size, bias=False) for _ in range(num_experts)]
        )
        self.neurons1 = nn.ModuleList([LIFNeuron(threshold=threshold, decay=decay) for _ in range(num_experts)])
        self.neurons2 = nn.ModuleList([LIFNeuron(threshold=threshold, decay=decay) for _ in range(num_experts)])
        self.neurons3 = nn.ModuleList([LIFNeuron(threshold=threshold, decay=decay) for _ in range(num_experts)])
        self._init_weights()
        for p in self.parameters():
            p.requires_grad = False

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.05)

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            return x.view(x.shape[0], -1)
        return x

    def set_gates(self, stage: str, weights: torch.Tensor) -> None:
        assert stage in self.gates, "stage must be one of stage1, stage2, stage3"
        w = weights.float()
        w = w / (w.sum() + 1e-8)
        self.gates[stage].data.copy_(w)

    def get_gates(self) -> dict:
        return {k: v.detach().clone() for k, v in self.gates.items()}

    def forward(self, spike_seq: torch.Tensor) -> tuple:
        """
        Args:
            spike_seq: (T, B, N_in) or (T, B, C, H, W)
        Returns:
            h_vis_low, h_vis_mid, h_vis_high: time-averaged combined outputs per stage.
        """
        T = spike_seq.shape[0]
        B = spike_seq.shape[1]
        device = spike_seq.device

        # States per expert
        s1_states = [n.init_state((B, self.stage1[0].out_features), device=device) for n in self.neurons1]
        s2_states = [n.init_state((B, self.stage2[0].out_features), device=device) for n in self.neurons2]
        s3_states = [n.init_state((B, self.stage3[0].out_features), device=device) for n in self.neurons3]

        stage1_spikes = []
        stage2_spikes = []
        stage3_spikes = []

        with torch.no_grad():
            for t in range(T):
                x_t = self._flatten(spike_seq[t])

                s1_out = []
                for i in range(self.num_experts):
                    h_in = self.stage1[i](x_t)
                    h_spk, s1_states[i] = self.neurons1[i](h_in, s1_states[i])
                    s1_out.append(h_spk)
                s1_out_stack = torch.stack(s1_out, dim=0)  # (E,B,H1)
                g1 = self.gates["stage1"].view(self.num_experts, 1, 1).to(device)
                s1_combined = (g1 * s1_out_stack).sum(dim=0)
                stage1_spikes.append(s1_combined)

                s2_out = []
                for i in range(self.num_experts):
                    mid_in = self.stage2[i](s1_combined)
                    mid_spk, s2_states[i] = self.neurons2[i](mid_in, s2_states[i])
                    s2_out.append(mid_spk)
                s2_out_stack = torch.stack(s2_out, dim=0)
                g2 = self.gates["stage2"].view(self.num_experts, 1, 1).to(device)
                s2_combined = (g2 * s2_out_stack).sum(dim=0)
                stage2_spikes.append(s2_combined)

                s3_out = []
                for i in range(self.num_experts):
                    high_in = self.stage3[i](s2_combined)
                    high_spk, s3_states[i] = self.neurons3[i](high_in, s3_states[i])
                    s3_out.append(high_spk)
                s3_out_stack = torch.stack(s3_out, dim=0)
                g3 = self.gates["stage3"].view(self.num_experts, 1, 1).to(device)
                s3_combined = (g3 * s3_out_stack).sum(dim=0)
                stage3_spikes.append(s3_combined)

        stage1_tensor = torch.stack(stage1_spikes, dim=0)  # (T,B,H1)
        stage2_tensor = torch.stack(stage2_spikes, dim=0)  # (T,B,H2)
        stage3_tensor = torch.stack(stage3_spikes, dim=0)  # (T,B,H3)

        h_vis_low = stage1_tensor.mean(dim=0)
        h_vis_mid = stage2_tensor.mean(dim=0)
        h_vis_high = stage3_tensor.mean(dim=0)
        return h_vis_low, h_vis_mid, h_vis_high
