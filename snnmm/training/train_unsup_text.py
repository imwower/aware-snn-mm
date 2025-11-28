import argparse
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.datasets.cifar10 import CIFAR10Dataset
from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.layers.stdp import stdp_update_linear
from snnmm.models.text_path import LabelSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised STDP training for label/text SNN.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--a-plus", type=float, default=0.01)
    parser.add_argument("--a-minus", type=float, default=0.01)
    parser.add_argument("--tau-pre", type=float, default=20.0)
    parser.add_argument("--tau-post", type=float, default=20.0)
    parser.add_argument("--high-rate", type=float, default=0.9)
    parser.add_argument("--low-rate", type=float, default=0.01)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | mps | cpu; auto prefers CUDA, then MPS on Apple Silicon, else CPU.",
    )
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument("--use-third-layer", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--semantic-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=1.0, help="LIF threshold for LabelSNN.")
    parser.add_argument("--dataset-name", type=str, default="cifar100", help="cifar100 or cifar10")
    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if key == "dataset" and isinstance(v, dict):
                if "name" in v:
                    args.dataset_name = v["name"]
                if "root" in v:
                    args.data_root = v["root"]
            elif hasattr(args, key):
                setattr(args, key, v)
    return args


def choose_device(device_flag: str) -> torch.device:
    if device_flag != "auto":
        return torch.device(device_flag)
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
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )


def init_traces(model: LabelSNN, device: torch.device) -> Dict[str, torch.Tensor]:
    traces: Dict[str, torch.Tensor] = {}
    traces["pre1"] = torch.zeros(model.fc1.in_features, device=device)
    traces["post1"] = torch.zeros(model.fc1.out_features, device=device)
    traces["pre2"] = torch.zeros(model.fc2.in_features, device=device)
    traces["post2"] = torch.zeros(model.fc2.out_features, device=device)
    if model.use_third:
        traces["pre3"] = torch.zeros(model.fc3.in_features, device=device)
        traces["post3"] = torch.zeros(model.fc3.out_features, device=device)
    return traces


def train_epoch(model: LabelSNN, dataloader: DataLoader, device: torch.device, args: argparse.Namespace) -> None:
    traces = init_traces(model, device)
    for step, batch in enumerate(dataloader, start=1):
        if args.limit_steps is not None and step > args.limit_steps:
            break
        _, labels, _ = batch
        labels = labels.to(device)
        spikes = label_poisson_encode(
            labels,
            timesteps=args.timesteps,
            n_labels=model.fc1.in_features,
            high_rate=args.high_rate,
            low_rate=args.low_rate,
            device=device,
        )

        s1 = model.neuron1.init_state((labels.shape[0], model.fc1.out_features), device=device)
        s2 = model.neuron2.init_state((labels.shape[0], model.fc2.out_features), device=device)
        s3 = model.neuron3.init_state((labels.shape[0], model.fc3.out_features), device=device) if model.use_third else None

        spk1_all = []
        spk2_all = []
        spk3_all = []
        upd_stats = []

        for t in range(args.timesteps):
            x_t = spikes[t]
            h1_in = model.fc1(x_t)
            spk1, s1 = model.neuron1(h1_in, s1)
            traces["pre1"], traces["post1"], upd1 = stdp_update_linear(
                model.fc1.weight,
                pre_spikes=x_t,
                post_spikes=spk1,
                pre_trace=traces["pre1"],
                post_trace=traces["post1"],
                lr=args.lr,
                tau_pre=args.tau_pre,
                tau_post=args.tau_post,
                a_plus=args.a_plus,
                a_minus=args.a_minus,
            )
            upd_stats.append(upd1)
            h2_in = model.fc2(spk1)
            spk2, s2 = model.neuron2(h2_in, s2)
            traces["pre2"], traces["post2"], upd2 = stdp_update_linear(
                model.fc2.weight,
                pre_spikes=spk1,
                post_spikes=spk2,
                pre_trace=traces["pre2"],
                post_trace=traces["post2"],
                lr=args.lr,
                tau_pre=args.tau_pre,
                tau_post=args.tau_post,
                a_plus=args.a_plus,
                a_minus=args.a_minus,
            )
            upd_stats.append(upd2)
            spk1_all.append(spk1)
            spk2_all.append(spk2)
            if model.use_third:
                h3_in = model.fc3(spk2)
                spk3, s3 = model.neuron3(h3_in, s3)
                traces["pre3"], traces["post3"], upd3 = stdp_update_linear(
                    model.fc3.weight,
                    pre_spikes=spk2,
                    post_spikes=spk3,
                    pre_trace=traces["pre3"],
                    post_trace=traces["post3"],
                    lr=args.lr,
                    tau_pre=args.tau_pre,
                    tau_post=args.tau_post,
                    a_plus=args.a_plus,
                    a_minus=args.a_minus,
                )
                upd_stats.append(upd3)
                spk3_all.append(spk3)

        spk1_rate = torch.stack(spk1_all, dim=0).mean().item() if spk1_all else 0.0
        spk2_rate = torch.stack(spk2_all, dim=0).mean().item() if spk2_all else 0.0
        spk3_rate = torch.stack(spk3_all, dim=0).mean().item() if spk3_all else 0.0
        weight_mean = (
            model.fc1.weight.mean() + model.fc2.weight.mean() + (model.fc3.weight.mean() if model.use_third else 0.0)
        ) / (3 if model.use_third else 2)
        upd_mean = float(torch.tensor(upd_stats).mean().item()) if upd_stats else 0.0

        print(
            f"[LOG] TextSTDP Step {step}, rate1={spk1_rate:.4f}, rate2={spk2_rate:.4f}, rate3={spk3_rate:.4f}, "
            f"weight_mean={weight_mean:.4f}, stdp_update_norm={upd_mean:.6f}"
        )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    n_labels = 10 if args.dataset_name.lower() == "cifar10" else 100
    model = LabelSNN(
        n_labels=n_labels,
        hidden_size=args.hidden_size,
        semantic_size=args.semantic_size,
        use_third=args.use_third_layer,
        threshold=args.threshold,
    ).to(device)
    dataloader = prepare_dataloader(args)

    for epoch in range(1, args.epochs + 1):
        print(f"=== Text Epoch {epoch}/{args.epochs} ===")
        train_epoch(model, dataloader, device, args)


if __name__ == "__main__":
    main()
