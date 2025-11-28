import torch

from snnmm.training.train_rstdp_classifier import update_readout_three_factor_debug


def test_three_factor_updates_signs():
    weight = torch.zeros(2, 3)
    pre_trace = torch.zeros(3)
    post_trace = torch.zeros(2)

    # two samples, 3 pre-neurons
    pre_spikes = torch.tensor([[1.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    # class_spikes shape (T,B,C)
    class_spikes = torch.zeros(2, 2, 2)
    # targets: class0 needs more spikes, class1 should be zero
    target_spikes = torch.tensor([[5.0, 0.0], [5.0, 0.0]])

    pre_trace, post_trace, upd_norm = update_readout_three_factor_debug(
        weight, pre_trace, post_trace, pre_spikes, class_spikes, target_spikes, lr=0.1
    )
    # class 0 weights should increase, class 1 decrease or stay small
    assert weight[0].mean() > 0
    assert weight[1].mean() <= 0
    assert upd_norm > 0
