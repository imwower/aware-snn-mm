import torch

from snnmm.layers.growth import CoreGrowthManager
from snnmm.models.core import AwarenessCoreSNN


def test_growth_and_prune():
    core = AwarenessCoreSNN(vis_dim=4, text_dim=4, core_dim=4, num_classes=3, threshold=0.0, decay=1.0)
    manager = CoreGrowthManager(
        init_dim=4,
        grow_buffer=2,
        grow_batch=2,
        grow_surprise_thresh=0.1,
        max_dim=16,
        prune_activity_thresh=0.1,
        prune_min_dim=2,
    )
    core.attach_growth_manager(manager)

    # accumulate high surprise samples to trigger growth
    for _ in range(3):
        z = torch.rand(2, core.core_dim)
        surprise = torch.ones(2) * 0.9
        manager.accumulate_sample_stats(z, surprise)
    grew = manager.maybe_grow(core)
    assert grew
    assert core.core_dim > 4
    # ensure new weights are not all zero in recurrent
    assert torch.any(core.recurrent.weight != 0)

    # simulate inactivity to prune
    manager.activity_sum = torch.zeros(core.core_dim)
    manager.activity_count = torch.ones(core.core_dim)
    pruned = manager.maybe_prune(core)
    assert pruned
    assert core.core_dim >= manager.prune_min_dim or core.core_dim < 4 + manager.grow_batch
