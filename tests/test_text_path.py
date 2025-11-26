import torch

from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.layers.stdp import stdp_update_linear
from snnmm.models.text_path import LabelSNN


def test_label_encoding_high_rate():
    labels = torch.tensor([5, 5])
    spikes = label_poisson_encode(labels, timesteps=10, n_labels=10, high_rate=0.9, low_rate=0.0)
    mean_rates = spikes.mean(dim=0)  # (B, n_labels)
    high_dim = mean_rates[:, 5].mean()
    other_dim = mean_rates[:, [i for i in range(10) if i != 5]].mean()
    assert high_dim > other_dim + 0.3


def test_label_snn_forward_and_stdp():
    model = LabelSNN(n_labels=10, hidden_size=6, semantic_size=4, threshold=0.0, decay=1.0)
    model.fc1.weight.data.fill_(0.5)
    model.fc2.weight.data.fill_(0.5)
    labels = torch.tensor([1, 2])
    spikes = label_poisson_encode(labels, timesteps=5, n_labels=10, high_rate=0.8, low_rate=0.0)
    h_low, h_sem = model(spikes)
    assert h_low.shape == (2, 6)
    assert h_sem.shape == (2, 4)
    assert h_low.sum() > 0
    assert h_sem.sum() > 0

    weight_before = model.fc1.weight.clone()
    pre_trace = torch.zeros(model.fc1.in_features)
    post_trace = torch.zeros(model.fc1.out_features)
    # pre spikes first
    pre_trace, post_trace, _ = stdp_update_linear(
        model.fc1.weight,
        pre_spikes=torch.tensor([[1.0] + [0.0] * 9]),
        post_spikes=torch.zeros(1, model.fc1.out_features),
        pre_trace=pre_trace,
        post_trace=post_trace,
        lr=1.0,
    )
    # post spikes later
    pre_trace, post_trace, _ = stdp_update_linear(
        model.fc1.weight,
        pre_spikes=torch.zeros(1, model.fc1.in_features),
        post_spikes=torch.ones(1, model.fc1.out_features),
        pre_trace=pre_trace,
        post_trace=post_trace,
        lr=1.0,
    )
    assert not torch.allclose(weight_before, model.fc1.weight)
