import torch

from snnmm.models.vision_path import VisionMultiStageSNN


def test_multistage_shapes_and_activity():
    model = VisionMultiStageSNN(
        input_size=12, stage1_size=10, stage2_size=8, stage3_size=6, num_experts=8, threshold=0.0, decay=1.0
    )
    for layer_group in (model.stage1, model.stage2, model.stage3):
        for layer in layer_group:
            layer.weight.data.fill_(0.5)
    # random binary spikes (T=5, B=2, N=12)
    spikes = (torch.rand(5, 2, 12) > 0.5).float()
    h_low, h_mid, h_high = model(spikes)
    assert h_low.shape == (2, 10)
    assert h_mid.shape == (2, 8)
    assert h_high.shape == (2, 6)
    assert h_low.sum() > 0
    assert h_mid.sum() > 0
    assert h_high.sum() > 0


def test_gates_normalized():
    model = VisionMultiStageSNN()
    rand_w = torch.rand(8)
    model.set_gates("stage1", rand_w)
    gates = model.get_gates()
    assert "stage1" in gates
    assert gates["stage1"].shape[0] == 8
    assert torch.isclose(gates["stage1"].sum(), torch.tensor(1.0), atol=1e-4)
