#!/usr/bin/env python
import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_raster(spikes: np.ndarray, title: str, save_path: str) -> None:
    """
    spikes: shape (T, N)
    """
    T, N = spikes.shape
    t_idx, n_idx = np.nonzero(spikes)
    plt.figure(figsize=(8, 4))
    plt.scatter(t_idx, n_idx, s=2, c="black")
    plt.xlabel("Time step")
    plt.ylabel("Neuron idx")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[LOG] Saved spike raster plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize spike raster from npy or random demo.")
    parser.add_argument("--spike-file", type=str, default=None, help="Path to npy file of spikes (T,N).")
    parser.add_argument("--save-path", type=str, default="logs/spikes_raster.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.spike_file and os.path.exists(args.spike_file):
        spikes = np.load(args.spike_file)
    else:
        # demo random spikes
        T, N = 100, 64
        spikes = (np.random.rand(T, N) < 0.05).astype(np.float32)
    plot_raster(spikes, "Spike Raster", args.save_path)


if __name__ == "__main__":
    main()
