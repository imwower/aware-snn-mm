import argparse
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.layers.gating import update_gates
from snnmm.layers.growth import CoreGrowthManager
from snnmm.models.core import AwarenessCoreSNN
from snnmm.models.text_path import LabelSNN
from snnmm.models.vision_path import VisionMultiStageSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align vision/text representations via awareness core (STDP/R-STDP-lite).")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--cycle-length", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--surprise-alpha", type=float, default=0.7)
    parser.add_argument("--surprise-beta", type=float, default=0.3)
    parser.add_argument("--growth-buffer", type=int, default=32)
    parser.add_argument("--growth-batch", type=int, default=8)
    parser.add_argument("--grow-surprise-thresh", type=float, default=0.6)
    parser.add_argument("--prune-activity-thresh", type=float, default=1e-3)
    parser.add_argument("--prune-min-dim", type=int, default=64)
    parser.add_argument("--max-core-dim", type=int, default=256)
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)
    return args


def choose_device(flag: str) -> torch.device:
    if flag != "auto":
        return torch.device(flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = CIFAR100Dataset(root=args.data_root, train=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)


def train_epoch(
    vision: VisionMultiStageSNN,
    text_model: LabelSNN,
    core: AwarenessCoreSNN,
    growth: CoreGrowthManager,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    for step, batch in enumerate(dataloader, start=1):
        if args.limit_steps is not None and step > args.limit_steps:
            break
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        vis_spikes = poisson_encode(images, timesteps=args.timesteps, max_rate=1.0, flatten=True)
        text_spikes = label_poisson_encode(labels, timesteps=args.timesteps, n_labels=100, device=device)

        h_low, h_mid, h_high = vision(vis_spikes)
        _, h_text_sem = text_model(text_spikes)

        z_core, z_vis, z_text, cls_spikes = core(h_high, h_text_sem, timesteps=args.timesteps, cycle_length=args.cycle_length)
        S = core.compute_surprise(cls_spikes, labels, z_vis, z_text, alpha=args.surprise_alpha, beta=args.surprise_beta)
        growth.accumulate_sample_stats(z_core, S)

        # Gate update suggestion for stage3
        delta_g = core.suggest_gate_delta(S, num_experts=vision.num_experts)
        new_g = update_gates(vision.gates["stage3"], delta_g.to(device))
        vision.gates["stage3"].data.copy_(new_g)

        # Optionally grow/prune every few steps
        grew, pruned = False, False
        if step % 50 == 0:
            grew = growth.maybe_grow(core)
            pruned = growth.maybe_prune(core)

        if step % 10 == 0:
            mean_S = S.mean().item()
            print(
                f"[LOG] Align Step {step}, mean_S={mean_S:.3f}, g_stage3={vision.gates['stage3'][:4].tolist()}..., "
                f"grew={grew}, pruned={pruned}, core_dim={core.core_dim}"
            )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    vision = VisionMultiStageSNN().to(device)
    text_model = LabelSNN().to(device)
    core = AwarenessCoreSNN(vis_dim=vision.stage3[0].out_features, text_dim=text_model.fc2.out_features).to(device)
    growth = CoreGrowthManager(
        init_dim=core.core_dim,
        grow_buffer=args.growth_buffer,
        grow_batch=args.growth_batch,
        grow_surprise_thresh=args.grow_surprise_thresh,
        prune_activity_thresh=args.prune_activity_thresh,
        prune_min_dim=args.prune_min_dim,
        max_dim=args.max_core_dim,
    )
    core.attach_growth_manager(growth)

    dataloader = prepare_dataloader(args)
    for epoch in range(1, args.epochs + 1):
        print(f"=== Align Core Epoch {epoch}/{args.epochs} ===")
        train_epoch(vision, text_model, core, growth, dataloader, device, args)


if __name__ == "__main__":
    main()
