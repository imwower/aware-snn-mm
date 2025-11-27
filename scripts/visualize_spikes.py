#!/usr/bin/env python
import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.models.core import AwarenessCoreSNN
from snnmm.models.text_path import LabelSNN
from snnmm.models.vision_path import VisionMultiStageSNN
from snnmm.datasets.cifar100 import CIFAR100Dataset
from torch.utils.data import DataLoader


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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[LOG] Saved spike raster plot to {save_path}")


def choose_device(flag: str) -> torch.device:
    if flag != "auto":
        return torch.device(flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(ckpt_path: str, device: torch.device):
    vision = VisionMultiStageSNN().to(device)
    text_model = LabelSNN().to(device)
    core = AwarenessCoreSNN(vis_dim=vision.stage3[0].out_features, text_dim=text_model.fc2.out_features).to(device)
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if "vision" in ckpt:
            vision.load_state_dict(ckpt["vision"])
        if "text_model" in ckpt:
            text_model.load_state_dict(ckpt["text_model"])
        if "core" in ckpt:
            core.load_state_dict(ckpt["core"])
        if "gates" in ckpt:
            for k, v in ckpt["gates"].items():
                vision.gates[k].data.copy_(v.to(device))
        print(f"[LOG] Loaded checkpoint {ckpt_path}")
    else:
        print("[LOG] No checkpoint provided, using random weights.")
    return vision, text_model, core


def collect_spikes_from_models(
    vision,
    text_model,
    core,
    data_root: str,
    timesteps: int,
    num_samples: int,
    device: torch.device,
    target_layer: str,
) -> np.ndarray:
    dataset = CIFAR100Dataset(root=data_root, train=False)
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    images, labels, _ = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    vis_spikes = poisson_encode(images, timesteps=timesteps, max_rate=1.0, flatten=True)
    text_spikes = label_poisson_encode(labels, timesteps=timesteps, n_labels=100, device=device)

    h_low, h_mid, h_high = vision(vis_spikes)
    _, h_text_sem = text_model(text_spikes)
    z_core, z_vis, z_text, cls_spikes = core(h_high, h_text_sem, timesteps=timesteps, cycle_length=10)

    if target_layer == "vision_stage1":
        return vis_spikes.cpu().numpy().reshape(timesteps, -1)
    if target_layer == "vision_stage3":
        return h_high.unsqueeze(0).repeat(timesteps, 1, 1).cpu().numpy().reshape(timesteps, -1)
    if target_layer == "core":
        return z_core.unsqueeze(0).repeat(timesteps, 1, 1).cpu().numpy().reshape(timesteps, -1)
    if target_layer == "classifier":
        return cls_spikes.cpu().numpy().reshape(timesteps, -1)
    return cls_spikes.cpu().numpy().reshape(timesteps, -1)


def main():
    parser = argparse.ArgumentParser(description="Visualize spike raster from npy or model forward.")
    parser.add_argument("--spike-file", type=str, default=None, help="Path to npy file of spikes (T,N).")
    parser.add_argument("--save-path", type=str, default="logs/spikes_raster.png")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint for forward visualization.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--num-samples", type=int, default=8, help="Samples to draw for forward pass.")
    parser.add_argument(
        "--target-layer",
        type=str,
        default="classifier",
        choices=["vision_stage1", "vision_stage3", "core", "classifier"],
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.spike_file and os.path.exists(args.spike_file):
        spikes = np.load(args.spike_file)
    else:
        device = choose_device(args.device)
        vision, text_model, core = load_models(args.checkpoint, device)
        spikes = collect_spikes_from_models(
            vision,
            text_model,
            core,
            data_root=args.data_root,
            timesteps=args.timesteps,
            num_samples=args.num_samples,
            device=device,
            target_layer=args.target_layer,
        )
    plot_raster(spikes, f"Spike Raster ({args.target_layer})", args.save_path)


if __name__ == "__main__":
    main()
