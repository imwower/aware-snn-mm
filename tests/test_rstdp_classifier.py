import torch

from snnmm.layers.stdp import rstdp_update_linear


def test_rstdp_positive_reward_increases_correct_weight():
    weight = torch.zeros(3, 2)
    elig = torch.zeros_like(weight)
    pre_trace = torch.zeros(2)
    post_trace = torch.zeros(3)

    # simulate correct class (index 1) firing with pre spikes active
    pre = torch.tensor([[1.0, 0.0]])
    post = torch.tensor([[0.0, 1.0, 0.0]])

    elig, pre_trace, post_trace, _ = rstdp_update_linear(
        weight, elig, pre_trace, post_trace, pre, post, reward=1.0, lr=0.5, a_minus=0.0
    )
    assert weight[1, 0] > 0


def test_rstdp_negative_reward_does_not_increase():
    weight = torch.zeros(3, 2)
    elig = torch.zeros_like(weight)
    pre_trace = torch.zeros(2)
    post_trace = torch.zeros(3)

    pre = torch.tensor([[1.0, 0.0]])
    post = torch.tensor([[0.0, 1.0, 0.0]])

    elig, pre_trace, post_trace, _ = rstdp_update_linear(
        weight, elig, pre_trace, post_trace, pre, post, reward=-1.0, lr=0.5
    )
    assert weight[1, 0] <= 0
