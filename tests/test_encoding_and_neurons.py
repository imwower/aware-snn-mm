import torch

from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.layers.neurons import LIFNeuron


def test_poisson_encoding_shape_and_binary():
    images = torch.rand(4, 3, 32, 32)
    spikes = poisson_encode(images, timesteps=5, max_rate=0.5, flatten=True)
    assert spikes.shape == (5, 4, 32 * 32 * 3)
    assert torch.all((spikes == 0) | (spikes == 1))


def test_lif_neuron_fires_and_resets():
    neuron = LIFNeuron(threshold=1.0, decay=1.0, reset_value=0.0)
    state = None
    fired = False
    for _ in range(3):
        input_spike = torch.full((1, 1), 0.6)
        spike, state = neuron(input_spike, state)
        if spike.item() > 0:
            fired = True
            assert state.item() == 0.0  # reset after spike
    assert fired, "Neuron should fire at least once under sustained input."

