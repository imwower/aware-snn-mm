import argparse
import os
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.datasets.cifar10 import CIFAR10Dataset
from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.layers.gating import update_gates
from snnmm.layers.stdp import rstdp_update_linear
from snnmm.models.core import AwarenessCoreSNN
from snnmm.models.text_path import LabelSNN
from snnmm.models.vision_path import VisionMultiStageSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-modulated STDP training for CIFAR-100 classification.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--cycle-length", type=int, default=10)
    parser.add_argument("--lr-core", type=float, default=1e-3)
    parser.add_argument("--lr-cls", type=float, default=0.006)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--surprise-alpha", type=float, default=0.4)
    parser.add_argument("--surprise-beta", type=float, default=0.0)
    parser.add_argument("--threshold-core", type=float, default=0.1)
    parser.add_argument("--threshold-cls", type=float, default=0.1)
    parser.add_argument("--reward-scale", type=float, default=2.0)
    parser.add_argument("--freeze-gates", action="store_true", help="If set, keep gates uniform during R-STDP.")
    parser.add_argument("--vision-ckpt", type=str, default=None, help="Optional vision checkpoint to load.")
    parser.add_argument("--text-ckpt", type=str, default=None, help="Optional text checkpoint to load.")
    parser.add_argument("--core-ckpt", type=str, default=None, help="Optional core checkpoint to load.")
    parser.add_argument("--normalize-core", action="store_true", help="L2 normalize core spikes before classifier.")
    parser.add_argument(
        "--post-agg",
        type=str,
        default="last",
        choices=["last", "mean", "max"],
        help="How to aggregate classifier spikes for R-STDP.",
    )
    parser.add_argument("--dataset-name", type=str, default="cifar100", help="cifar100 or cifar10")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--target-spikes-pos", type=float, default=5.0)
    parser.add_argument("--target-spikes-neg", type=float, default=0.0)
    parser.add_argument("--debug-mode", type=str, default="normal")
    parser.add_argument("--use-three-factor-debug", action="store_true")
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if key in ("dataset", "rstdp", "train", "sim", "model") and isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    sub_key = sub_k.replace("-", "_")
                    if sub_key == "name" and key == "dataset":
                        args.dataset_name = sub_v
                    elif sub_key == "root" and key == "dataset":
                        args.data_root = sub_v
                    elif sub_key == "num_classes" and key == "model":
                        args.num_classes = sub_v
                    elif hasattr(args, sub_key):
                        setattr(args, sub_key, sub_v)
            elif hasattr(args, key):
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
    if args.dataset_name.lower() == "cifar10":
        dataset = CIFAR10Dataset(root=args.data_root, train=True)
    else:
        dataset = CIFAR100Dataset(root=args.data_root, train=True)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)


def init_traces(core: AwarenessCoreSNN, device: torch.device) -> Dict[str, torch.Tensor]:
    traces: Dict[str, torch.Tensor] = {}
    traces["pre_cls"] = torch.zeros(core.classifier.in_features, device=device)
    traces["post_cls"] = torch.zeros(core.classifier.out_features, device=device)
    traces["elig_cls"] = torch.zeros_like(core.classifier.weight, device=device)
    return traces


def compute_reward(pred: torch.Tensor, labels: torch.Tensor, reward_scale: float = 1.0) -> float:
    correct = (pred == labels).float()
    base_R = torch.where(correct > 0, torch.ones_like(correct), -torch.ones_like(correct))
    # 固定奖励，不再被惊讶度衰减
    return (base_R * reward_scale).mean().item()


def update_readout_three_factor_debug(
    weight: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    pre_spikes: torch.Tensor,
    class_spikes: torch.Tensor,
    target_spikes: torch.Tensor,
    lr: float,
    tau_pre: float = 20.0,
    tau_post: float = 20.0,
    clip: float = 1.0,
) -> tuple:
    """
    Three-factor rule for readout: error per neuron * pre_trace.
    weight: (C, H), pre_spikes: (B, H), class_spikes: (T,B,C), target_spikes: (B,C)
    """
    with torch.no_grad():
        pre_trace = pre_trace * (1.0 - 1.0 / tau_pre) + pre_spikes.mean(dim=0)
        post_trace = post_trace * (1.0 - 1.0 / tau_post) + class_spikes.mean(dim=(0, 1))
        n_spikes = class_spikes.sum(dim=0)  # (B,C)
        error = target_spikes - n_spikes  # (B,C)
        err_mean = error.mean(dim=0)  # (C,)
        update = lr * err_mean.view(-1, 1) * pre_trace.view(1, -1)
        weight += update
        weight.clamp_(-clip, clip)
        update_norm = update.norm().item()
    return pre_trace, post_trace, update_norm


def train_epoch(
    vision: VisionMultiStageSNN,
    text_model: LabelSNN,
    core: AwarenessCoreSNN,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    traces: Dict[str, torch.Tensor],
) -> None:
    total = 0
    correct = 0
    surprise_list = []
    num_classes = 10 if args.dataset_name.lower() == "cifar10" else 100
    for step, batch in enumerate(dataloader, start=1):
        if args.limit_steps is not None and step > args.limit_steps:
            break
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        total += labels.numel()

        # encode
        vis_spikes = poisson_encode(images, timesteps=args.timesteps, max_rate=1.0, flatten=True)
        text_spikes = label_poisson_encode(labels, timesteps=args.timesteps, n_labels=num_classes, device=device)

        # forward through vision and text
        h_low, h_mid, h_high = vision(vis_spikes)
        h_text_low, h_text_sem = text_model(text_spikes)

        # core forward
        z_core, z_vis, z_text, cls_spikes = core(h_high, h_text_sem, timesteps=args.timesteps, cycle_length=args.cycle_length)
        rates = cls_spikes.mean(dim=0)
        pred = rates.argmax(dim=1)
        correct += (pred == labels).sum().item()

        # surprise
        S = core.compute_surprise(cls_spikes, labels, z_vis, z_text, alpha=args.surprise_alpha, beta=args.surprise_beta)
        surprise_list.append(S.mean().item())

        if (
            args.dataset_name.lower() == "cifar10"
            and args.debug_mode == "readout_only"
            and args.use_three_factor_debug
        ):
            target = torch.full((labels.shape[0], num_classes), args.target_spikes_neg, device=device)
            target[torch.arange(labels.shape[0]), labels] = args.target_spikes_pos
            pre_spikes = z_core
            if args.normalize_core:
                pre_spikes = pre_spikes / (pre_spikes.norm(dim=1, keepdim=True) + 1e-8)
            traces["pre_cls"], traces["post_cls"], upd = update_readout_three_factor_debug(
                core.classifier.weight,
                traces["pre_cls"],
                traces["post_cls"],
                pre_spikes,
                cls_spikes,
                target,
                lr=args.lr_cls,
            )
            update_norm = upd
        else:
            # reward and R-STDP on classifier
            R = compute_reward(pred, labels, reward_scale=args.reward_scale)
            pre_spikes = z_core
            post_spikes = cls_spikes[-1]
            if args.normalize_core:
                pre_spikes = pre_spikes / (pre_spikes.norm(dim=1, keepdim=True) + 1e-8)
            if args.post_agg == "mean":
                post_spikes = cls_spikes.mean(dim=0)
            elif args.post_agg == "max":
                post_spikes = cls_spikes.max(dim=0).values

            traces["elig_cls"], traces["pre_cls"], traces["post_cls"], upd = rstdp_update_linear(
                core.classifier.weight,
                traces["elig_cls"],
                traces["pre_cls"],
                traces["post_cls"],
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                reward=R,
                lr=args.lr_cls,
            )
            update_norm = upd

        # gate updates suggestion (vision stage3 as example)
        if not args.freeze_gates:
            delta_g = core.suggest_gate_delta(S, num_experts=vision.num_experts, scale=0.2)
            new_g = update_gates(vision.gates["stage3"], delta_g.to(device))
            vision.gates["stage3"].data.copy_(new_g)

        if step % 5 == 0:
            acc = correct / total if total > 0 else 0.0
            mean_S = sum(surprise_list) / len(surprise_list)
            g3 = vision.gates["stage3"].detach().cpu().numpy().tolist()
            w_mean = core.classifier.weight.mean().item()
            w_std = core.classifier.weight.std().item()
            mean_spk = cls_spikes.sum(dim=0).mean().item()
            log_prefix = "CIFAR10-RSTDP" if args.dataset_name.lower() == "cifar10" else "RSTDP"
            print(
                f"[LOG] {log_prefix} Ep step {step}, acc={acc:.3f}, mean_S={mean_S:.3f}, "
                f"mean_spikes={mean_spk:.3f}, w_mean={w_mean:.4f}, w_std={w_std:.4f}, g_stage3={g3[:4]}..."
            )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    num_classes = 10 if args.dataset_name.lower() == "cifar10" else args.num_classes
    vision = VisionMultiStageSNN().to(device)
    text_model = LabelSNN(n_labels=num_classes).to(device)

    # Load optional checkpoints
    if args.vision_ckpt and os.path.exists(args.vision_ckpt):
        ckpt_v = torch.load(args.vision_ckpt, map_location=device)
        if "vision" in ckpt_v:
            vision.load_state_dict(ckpt_v["vision"])
            if "gates" in ckpt_v:
                for k, v in ckpt_v["gates"].items():
                    vision.gates[k].data.copy_(v.to(device))
        print(f"[LOG] Loaded vision ckpt {args.vision_ckpt}")

    if args.text_ckpt and os.path.exists(args.text_ckpt):
        ckpt_t = torch.load(args.text_ckpt, map_location=device)
        if "text_model" in ckpt_t:
            text_model.load_state_dict(ckpt_t["text_model"])
        print(f"[LOG] Loaded text ckpt {args.text_ckpt}")

    # Build core with possible custom core_dim from ckpt
    core_dim_override = None
    if args.core_ckpt and os.path.exists(args.core_ckpt):
        ckpt_c = torch.load(args.core_ckpt, map_location=device)
        if "core" in ckpt_c and "classifier.weight" in ckpt_c["core"]:
            core_dim_override = ckpt_c["core"]["classifier.weight"].shape[1]
    core = AwarenessCoreSNN(
        vis_dim=vision.stage3[0].out_features,
        text_dim=text_model.fc2.out_features,
        core_dim=core_dim_override or 192,
        num_classes=num_classes,
        threshold_core=args.threshold_core,
        threshold_cls=args.threshold_cls,
    ).to(device)
    if args.core_ckpt and os.path.exists(args.core_ckpt):
        if "core" in ckpt_c:
            core.load_state_dict(ckpt_c["core"])
        print(f"[LOG] Loaded core ckpt {args.core_ckpt}")

    dataloader = prepare_dataloader(args)
    traces = init_traces(core, device)

    import os
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"=== R-STDP Epoch {epoch}/{args.epochs} ===")
        train_epoch(vision, text_model, core, dataloader, device, args, traces)
        ckpt_path = os.path.join("checkpoints", f"rstdp_epoch{epoch}.pt")
        torch.save(
            {
                "vision": vision.state_dict(),
                "text_model": text_model.state_dict(),
                "core": core.state_dict(),
                "gates": {k: v.detach().cpu() for k, v in vision.get_gates().items()},
                "epoch": epoch,
            },
            ckpt_path,
        )
        print(f"[LOG] Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
