from typing import Tuple

import torch
import torch.nn.functional as F


def _decay(trace: torch.Tensor, tau: float) -> torch.Tensor:
    decay = 1.0 - 1.0 / max(tau, 1.0)
    return trace * decay


def stdp_update_linear(
    weight: torch.Tensor,
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    lr: float = 1e-3,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
    a_plus: float = 0.01,
    a_minus: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    STDP update for a linear layer weight matrix (out, in).

    Args:
        weight: weight tensor to update in-place.
        pre_spikes: (B, in) binary input spikes.
        post_spikes: (B, out) binary output spikes.
        pre_trace/post_trace: 1D tensors storing traces.
    Returns:
        Updated pre_trace, post_trace, and update norm.
    """
    with torch.no_grad():
        pre_trace = _decay(pre_trace, tau_pre) + pre_spikes.float().mean(dim=0)
        post_trace = _decay(post_trace, tau_post) + post_spikes.float().mean(dim=0)

        dw_plus = a_plus * torch.outer(post_spikes.float().mean(dim=0), pre_trace)
        dw_minus = a_minus * torch.outer(post_trace, pre_spikes.float().mean(dim=0))
        dw = dw_plus - dw_minus
        weight += lr * dw
        update_norm = dw.norm().item()
    return pre_trace, post_trace, update_norm


def stdp_update_conv(
    weight: torch.Tensor,
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    lr: float = 1e-3,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
    a_plus: float = 0.01,
    a_minus: float = 0.01,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    STDP update for convolutional weights (out, in, k, k).
    Uses unfolded patches for a simple Hebbian approximation.
    """
    with torch.no_grad():
        kh, kw = weight.shape[2], weight.shape[3]
        patches = F.unfold(pre_spikes.float(), kernel_size=(kh, kw), stride=stride, padding=padding)
        post_flat = post_spikes.float().view(post_spikes.shape[0], post_spikes.shape[1], -1)

        pre_trace = _decay(pre_trace, tau_pre) + patches.mean(dim=(0, 2))
        post_trace = _decay(post_trace, tau_post) + post_flat.mean(dim=(0, 2))

        pre_mean = patches.mean(dim=(0, 2)).view(weight.shape[1], kh, kw)
        post_mean = post_flat.mean(dim=(0, 2))  # shape (out,)

        dw_plus = a_plus * post_mean[:, None, None, None] * pre_trace.view(weight.shape[1], kh, kw)
        dw_minus = a_minus * post_trace[:, None, None, None] * pre_mean
        dw = dw_plus - dw_minus
        weight += lr * dw
        update_norm = dw.norm().item()
    return pre_trace, post_trace, update_norm
