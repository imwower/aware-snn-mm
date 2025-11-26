from typing import List, Optional

import torch


def _expand_linear_out(layer: torch.nn.Linear, add: int, init_rows: torch.Tensor) -> torch.nn.Linear:
    """Expand a Linear layer by adding output rows."""
    old_w = layer.weight.data
    new_w = torch.cat([old_w, init_rows], dim=0)
    new_layer = torch.nn.Linear(layer.in_features, layer.out_features + add, bias=False)
    new_layer.weight.data.copy_(new_w)
    new_layer.weight.requires_grad = False
    return new_layer


def _expand_linear_out_in(layer: torch.nn.Linear, add: int, init_block: torch.Tensor) -> torch.nn.Linear:
    """Expand a square-ish Linear layer on both out and in dims (for recurrent)."""
    old_w = layer.weight.data
    out, inp = old_w.shape
    top = torch.cat([old_w, torch.zeros(out, add, device=old_w.device)], dim=1)
    bottom = torch.cat([torch.zeros(add, inp, device=old_w.device), init_block], dim=1)
    new_w = torch.cat([top, bottom], dim=0)
    new_layer = torch.nn.Linear(layer.in_features + add, layer.out_features + add, bias=False)
    new_layer.weight.data.copy_(new_w)
    new_layer.weight.requires_grad = False
    return new_layer


def _shrink_linear(layer: torch.nn.Linear, keep_idx: torch.Tensor) -> torch.nn.Linear:
    """Keep only rows/columns for a square-ish linear used as recurrent."""
    old_w = layer.weight.data
    w = old_w[keep_idx][:, keep_idx]
    new_layer = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
    new_layer.weight.data.copy_(w)
    new_layer.weight.requires_grad = False
    return new_layer


class CoreGrowthManager:
    """
    Manage growth and pruning of core neurons based on surprise and activity.
    Simplified heuristic: buffer high-surprise activations; grow when buffer full; prune low-activity neurons.
    """

    def __init__(
        self,
        init_dim: int,
        grow_buffer: int = 16,
        grow_batch: int = 4,
        grow_surprise_thresh: float = 0.6,
        prune_activity_thresh: float = 1e-3,
        prune_min_dim: int = 32,
        max_dim: int = 512,
    ) -> None:
        self.grow_buffer = grow_buffer
        self.grow_batch = grow_batch
        self.grow_surprise_thresh = grow_surprise_thresh
        self.prune_activity_thresh = prune_activity_thresh
        self.prune_min_dim = prune_min_dim
        self.max_dim = max_dim

        self.buffer: List[torch.Tensor] = []
        self.activity_sum = torch.zeros(init_dim)
        self.activity_count = torch.zeros(init_dim)

    def accumulate_sample_stats(self, z_core: torch.Tensor, surprise: torch.Tensor) -> None:
        """
        z_core: (B, D) core activations (rates)
        surprise: (B,)
        """
        z_core = z_core.detach().cpu()
        surprise = surprise.detach().cpu()
        high_mask = surprise > self.grow_surprise_thresh
        if high_mask.any():
            self.buffer.append(z_core[high_mask])
            if len(self.buffer) > self.grow_buffer:
                self.buffer.pop(0)
        self.activity_sum += z_core.mean(dim=0)
        self.activity_count += torch.ones_like(self.activity_sum)

    def maybe_grow(self, core_model) -> bool:
        """
        Expand core_model (AwarenessCoreSNN) by adding neurons.
        Returns True if growth happened.
        """
        current_dim = core_model.core_dim
        if current_dim >= self.max_dim or len(self.buffer) < self.grow_buffer:
            return False
        grow_n = min(self.grow_batch, self.max_dim - current_dim)
        device = next(core_model.parameters()).device
        # initialize new rows for projections with small random
        init_vis_rows = torch.randn(grow_n, core_model.vis_proj.in_features, device=device) * 0.01
        init_txt_rows = torch.randn(grow_n, core_model.text_proj.in_features, device=device) * 0.01
        # Expand projections
        core_model.vis_proj = _expand_linear_out(core_model.vis_proj, grow_n, init_vis_rows)
        core_model.text_proj = _expand_linear_out(core_model.text_proj, grow_n, init_txt_rows)
        # Expand recurrent
        init_block = torch.eye(grow_n, device=device) * 0.01
        core_model.recurrent = _expand_linear_out_in(core_model.recurrent, grow_n, init_block)
        # Expand classifier columns
        old_cls = core_model.classifier.weight.data
        new_cls = torch.cat([old_cls, torch.zeros(old_cls.shape[0], grow_n, device=device)], dim=1)
        new_classifier = torch.nn.Linear(core_model.core_dim + grow_n, core_model.num_classes, bias=False)
        new_classifier.weight.data.copy_(new_cls)
        new_classifier.weight.requires_grad = False
        core_model.classifier = new_classifier

        # Update meta
        core_model.core_dim += grow_n
        self.activity_sum = torch.cat([self.activity_sum, torch.zeros(grow_n)])
        self.activity_count = torch.cat([self.activity_count, torch.zeros(grow_n)])
        self.buffer = []
        return True

    def maybe_prune(self, core_model) -> bool:
        """
        Prune low-activity neurons.
        Returns True if pruning happened.
        """
        if core_model.core_dim <= self.prune_min_dim:
            return False
        activity_mean = self.activity_sum / (self.activity_count + 1e-8)
        keep_mask = activity_mean >= self.prune_activity_thresh
        if keep_mask.sum().item() == core_model.core_dim:
            return False
        if keep_mask.sum().item() < self.prune_min_dim:
            # ensure minimum size
            topk = torch.topk(activity_mean, k=self.prune_min_dim).indices
            tmp_mask = torch.zeros_like(keep_mask)
            tmp_mask[topk] = True
            keep_mask = tmp_mask
        keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        device = next(core_model.parameters()).device
        keep_idx = keep_idx.to(device)

        # cache old weights before rebuild
        old_vis_w = core_model.vis_proj.weight.data.clone()
        old_txt_w = core_model.text_proj.weight.data.clone()
        old_rec = core_model.recurrent
        old_cls_w = core_model.classifier.weight.data.clone()

        core_model.vis_proj = torch.nn.Linear(old_vis_w.shape[1], keep_idx.numel(), bias=False)
        core_model.vis_proj.weight.data.copy_(old_vis_w[keep_idx])
        core_model.vis_proj.weight.requires_grad = False

        core_model.text_proj = torch.nn.Linear(old_txt_w.shape[1], keep_idx.numel(), bias=False)
        core_model.text_proj.weight.data.copy_(old_txt_w[keep_idx])
        core_model.text_proj.weight.requires_grad = False

        core_model.recurrent = _shrink_linear(old_rec, keep_idx)

        new_cls = torch.nn.Linear(keep_idx.numel(), core_model.num_classes, bias=False)
        new_cls.weight.data.copy_(old_cls_w[:, keep_idx])
        new_cls.weight.requires_grad = False
        core_model.classifier = new_cls

        core_model.core_dim = keep_idx.numel()
        self.activity_sum = self.activity_sum[keep_idx.cpu()]
        self.activity_count = self.activity_count[keep_idx.cpu()]
        return True
