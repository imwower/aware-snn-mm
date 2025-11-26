import torch

from snnmm.models.core import AwarenessCoreSNN


def test_core_forward_shapes_and_activity():
    core = AwarenessCoreSNN(vis_dim=6, text_dim=6, core_dim=12, num_classes=5, threshold=0.0, decay=1.0)
    h_vis = torch.rand(2, 6)
    h_text = torch.rand(2, 6)
    z_core, z_vis, z_text, cls_spikes = core(h_vis, h_text, timesteps=5, cycle_length=2)
    assert z_core.shape == (2, 12)
    assert z_vis.shape == (2, 12)
    assert z_text.shape == (2, 12)
    assert cls_spikes.shape == (5, 2, 5)
    assert cls_spikes.sum() > 0


def test_compute_surprise_monotonicity():
    core = AwarenessCoreSNN(vis_dim=4, text_dim=4, core_dim=6, num_classes=3)
    # Low surprise case: aligned and confident
    cls_spikes_good = torch.zeros(5, 1, 3)
    cls_spikes_good[:, 0, 1] = 5.0
    z_vis_good = torch.zeros(1, 6)
    z_text_good = torch.zeros(1, 6)
    S_good = core.compute_surprise(cls_spikes_good, torch.tensor([1]), z_vis_good, z_text_good)

    # High surprise: misaligned and low confidence
    cls_spikes_bad = torch.zeros(5, 1, 3)
    cls_spikes_bad[:, 0, :] = 1.0
    z_vis_bad = torch.ones(1, 6)
    z_text_bad = torch.zeros(1, 6)
    S_bad = core.compute_surprise(cls_spikes_bad, torch.tensor([2]), z_vis_bad, z_text_bad)

    assert S_bad.mean() > S_good.mean()
