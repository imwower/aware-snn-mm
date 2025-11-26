import torch

from snnmm.layers.stdp import stdp_update_linear


def test_stdp_weight_increases_when_pre_precedes_post():
    weight = torch.zeros(1, 2)
    pre_trace = torch.zeros(2)
    post_trace = torch.zeros(1)

    # Pre spikes first
    pre_trace, post_trace, _ = stdp_update_linear(
        weight, torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0]]), pre_trace, post_trace, lr=1.0
    )
    pre_trace, post_trace, _ = stdp_update_linear(
        weight, torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0]]), pre_trace, post_trace, lr=1.0
    )
    assert weight[0, 0] > 0


def test_stdp_weight_decreases_when_post_precedes_pre():
    weight = torch.zeros(1, 2)
    pre_trace = torch.zeros(2)
    post_trace = torch.zeros(1)

    # Post spikes first
    pre_trace, post_trace, _ = stdp_update_linear(
        weight, torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0]]), pre_trace, post_trace, lr=1.0
    )
    pre_trace, post_trace, _ = stdp_update_linear(
        weight, torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0]]), pre_trace, post_trace, lr=1.0
    )
    assert weight[0, 0] < 0

