import argparse
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.layers.stdp import stdp_update_linear
from snnmm.models.vision_path import VisionSNNStage1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised STDP pretraining for vision SNN.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--a-plus", type=float, default=0.01)
    parser.add_argument("--a-minus", type=float, default=0.01)
    parser.add_argument("--tau-pre", type=float, default=20.0)
    parser.add_argument("--tau-post", type=float, default=20.0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | mps | cpu; auto prefers CUDA, then MPS on Apple Silicon, else CPU.",
    )
    parser.add_argument("--limit-steps", type=int, default=None, help="Optional number of steps per epoch.")
    parser.add_argument("--timesteps-max-rate", type=float, default=1.0, help="Max spike rate for encoding.")
    parser.add_argument("--use-second-layer", action="store_true", help="Enable second LIF layer.")
    return parser.parse_args()


def prepare_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = CIFAR100Dataset(root=args.data_root, train=True)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def init_traces(model: VisionSNNStage1, device: torch.device) -> Dict[str, torch.Tensor]:
    traces: Dict[str, torch.Tensor] = {}
    traces["pre1"] = torch.zeros(model.fc1.in_features, device=device)
    traces["post1"] = torch.zeros(model.fc1.out_features, device=device)
    if model.use_second_layer:
        traces["pre2"] = torch.zeros(model.fc2.in_features, device=device)
        traces["post2"] = torch.zeros(model.fc2.out_features, device=device)
    return traces


def train_epoch(
    model: VisionSNNStage1,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    traces = init_traces(model, device)
    total_steps = 0
    for step, batch in enumerate(dataloader, start=1):
        if args.limit_steps is not None and step > args.limit_steps:
            break
        images, _, _ = batch
        images = images.to(device)

        spike_seq = poisson_encode(
            images, timesteps=args.timesteps, max_rate=args.timesteps_max_rate, flatten=True
        )

        h_state = model.neuron1.init_state((images.shape[0], model.fc1.out_features), device=device)
        o_state = model.neuron2.init_state((images.shape[0], model.fc2.out_features), device=device) if model.use_second_layer else None
        step_updates = []
        outputs = []

        for t in range(args.timesteps):
            x_t = spike_seq[t]
            h_input = model.fc1(x_t)
            h_spike, h_state = model.neuron1(h_input, h_state)
            traces["pre1"], traces["post1"], upd1 = stdp_update_linear(
                model.fc1.weight,
                pre_spikes=x_t,
                post_spikes=h_spike,
                pre_trace=traces["pre1"],
                post_trace=traces["post1"],
                lr=args.lr,
                tau_pre=args.tau_pre,
                tau_post=args.tau_post,
                a_plus=args.a_plus,
                a_minus=args.a_minus,
            )
            step_updates.append(upd1)

            if model.use_second_layer:
                o_input = model.fc2(h_spike)
                o_spike, o_state = model.neuron2(o_input, o_state)
                traces["pre2"], traces["post2"], upd2 = stdp_update_linear(
                    model.fc2.weight,
                    pre_spikes=h_spike,
                    post_spikes=o_spike,
                    pre_trace=traces["pre2"],
                    post_trace=traces["post2"],
                    lr=args.lr,
                    tau_pre=args.tau_pre,
                    tau_post=args.tau_post,
                    a_plus=args.a_plus,
                    a_minus=args.a_minus,
                )
                step_updates.append(upd2)
                outputs.append(o_spike)
            else:
                outputs.append(h_spike)

        output_tensor = torch.stack(outputs, dim=0)
        mean_rate = output_tensor.mean().item()
        weight_mean = model.fc1.weight.mean().item()
        if model.use_second_layer:
            weight_mean = (weight_mean + model.fc2.weight.mean().item()) / 2.0
        update_norm = float(torch.tensor(step_updates).mean().item())

        total_steps += 1
        print(
            f"[LOG] Epoch step {step}, mean_rate={mean_rate:.4f}, "
            f"weight_mean={weight_mean:.4f}, stdp_update_norm={update_norm:.6f}"
        )


def choose_device(device_flag: str) -> torch.device:
    if device_flag != "auto":
        return torch.device(device_flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = VisionSNNStage1(use_second_layer=args.use_second_layer).to(device)
    dataloader = prepare_dataloader(args)

    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        train_epoch(model, dataloader, device, args)


if __name__ == "__main__":
    main()
