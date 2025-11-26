import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.layers.stdp import stdp_update_linear
from snnmm.models.vision_path import VisionMultiStageSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised STDP pretraining for multi-stage vision SNN.")
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
    parser.add_argument("--stage-depth", type=int, default=1, choices=[1, 2, 3], help="Train up to stage depth.")
    parser.add_argument("--stage1-size", type=int, default=256)
    parser.add_argument("--stage2-size", type=int, default=256)
    parser.add_argument("--stage3-size", type=int, default=128)
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


def init_traces(model: VisionMultiStageSNN, device: torch.device) -> Dict[str, List[torch.Tensor]]:
    traces: Dict[str, List[torch.Tensor]] = {}
    traces["stage1_pre"] = [torch.zeros(model.stage1[0].in_features, device=device) for _ in range(model.num_experts)]
    traces["stage1_post"] = [torch.zeros(model.stage1[0].out_features, device=device) for _ in range(model.num_experts)]
    traces["stage2_pre"] = [torch.zeros(model.stage2[0].in_features, device=device) for _ in range(model.num_experts)]
    traces["stage2_post"] = [torch.zeros(model.stage2[0].out_features, device=device) for _ in range(model.num_experts)]
    traces["stage3_pre"] = [torch.zeros(model.stage3[0].in_features, device=device) for _ in range(model.num_experts)]
    traces["stage3_post"] = [torch.zeros(model.stage3[0].out_features, device=device) for _ in range(model.num_experts)]
    return traces


def choose_device(device_flag: str) -> torch.device:
    if device_flag != "auto":
        return torch.device(device_flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model: VisionMultiStageSNN,
    dataloader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    traces = init_traces(model, device)
    for step, batch in enumerate(dataloader, start=1):
        if args.limit_steps is not None and step > args.limit_steps:
            break
        images, _, _ = batch
        images = images.to(device)
        spikes_in = poisson_encode(images, timesteps=args.timesteps, max_rate=args.timesteps_max_rate, flatten=True)

        s1_states = [model.neurons1[i].init_state((images.shape[0], model.stage1[0].out_features), device=device) for i in range(model.num_experts)]
        s2_states = [model.neurons2[i].init_state((images.shape[0], model.stage2[0].out_features), device=device) for i in range(model.num_experts)]
        s3_states = [model.neurons3[i].init_state((images.shape[0], model.stage3[0].out_features), device=device) for i in range(model.num_experts)]

        stage1_spk_all = []
        stage2_spk_all = []
        stage3_spk_all = []
        step_updates = []

        for t in range(args.timesteps):
            x_t = spikes_in[t]

            # Stage 1 per expert
            s1_out = []
            for i in range(model.num_experts):
                h_in = model.stage1[i](x_t)
                h_spk, s1_states[i] = model.neurons1[i](h_in, s1_states[i])
                traces["stage1_pre"][i], traces["stage1_post"][i], upd = stdp_update_linear(
                    model.stage1[i].weight,
                    pre_spikes=x_t,
                    post_spikes=h_spk,
                    pre_trace=traces["stage1_pre"][i],
                    post_trace=traces["stage1_post"][i],
                    lr=args.lr,
                    tau_pre=args.tau_pre,
                    tau_post=args.tau_post,
                    a_plus=args.a_plus,
                    a_minus=args.a_minus,
                )
                step_updates.append(upd)
                s1_out.append(h_spk)
            s1_out_stack = torch.stack(s1_out, dim=0)
            g1 = model.gates["stage1"].view(model.num_experts, 1, 1).to(device)
            s1_combined = (g1 * s1_out_stack).sum(dim=0)
            stage1_spk_all.append(s1_combined)

            if args.stage_depth >= 2:
                s2_out = []
                for i in range(model.num_experts):
                    mid_in = model.stage2[i](s1_combined)
                    mid_spk, s2_states[i] = model.neurons2[i](mid_in, s2_states[i])
                    traces["stage2_pre"][i], traces["stage2_post"][i], upd = stdp_update_linear(
                        model.stage2[i].weight,
                        pre_spikes=s1_combined,
                        post_spikes=mid_spk,
                        pre_trace=traces["stage2_pre"][i],
                        post_trace=traces["stage2_post"][i],
                        lr=args.lr,
                        tau_pre=args.tau_pre,
                        tau_post=args.tau_post,
                        a_plus=args.a_plus,
                        a_minus=args.a_minus,
                    )
                    step_updates.append(upd)
                    s2_out.append(mid_spk)
                s2_out_stack = torch.stack(s2_out, dim=0)
                g2 = model.gates["stage2"].view(model.num_experts, 1, 1).to(device)
                s2_combined = (g2 * s2_out_stack).sum(dim=0)
                stage2_spk_all.append(s2_combined)
            else:
                s2_combined = None

            if args.stage_depth >= 3 and s2_combined is not None:
                s3_out = []
                for i in range(model.num_experts):
                    hi_in = model.stage3[i](s2_combined)
                    hi_spk, s3_states[i] = model.neurons3[i](hi_in, s3_states[i])
                    traces["stage3_pre"][i], traces["stage3_post"][i], upd = stdp_update_linear(
                        model.stage3[i].weight,
                        pre_spikes=s2_combined,
                        post_spikes=hi_spk,
                        pre_trace=traces["stage3_pre"][i],
                        post_trace=traces["stage3_post"][i],
                        lr=args.lr,
                        tau_pre=args.tau_pre,
                        tau_post=args.tau_post,
                        a_plus=args.a_plus,
                        a_minus=args.a_minus,
                    )
                    step_updates.append(upd)
                    s3_out.append(hi_spk)
                s3_out_stack = torch.stack(s3_out, dim=0)
                g3 = model.gates["stage3"].view(model.num_experts, 1, 1).to(device)
                s3_combined = (g3 * s3_out_stack).sum(dim=0)
                stage3_spk_all.append(s3_combined)

        stage1_rate = torch.stack(stage1_spk_all, dim=0).mean().item() if stage1_spk_all else 0.0
        stage2_rate = torch.stack(stage2_spk_all, dim=0).mean().item() if stage2_spk_all else 0.0
        stage3_rate = torch.stack(stage3_spk_all, dim=0).mean().item() if stage3_spk_all else 0.0

        weight_means = []
        weight_means.append(torch.stack([m.weight.mean() for m in model.stage1]).mean().item())
        if args.stage_depth >= 2:
            weight_means.append(torch.stack([m.weight.mean() for m in model.stage2]).mean().item())
        if args.stage_depth >= 3:
            weight_means.append(torch.stack([m.weight.mean() for m in model.stage3]).mean().item())
        weight_mean = sum(weight_means) / len(weight_means)
        update_norm = float(torch.tensor(step_updates).mean().item()) if step_updates else 0.0

        gates_info = {k: v.detach().cpu().numpy().tolist() for k, v in model.get_gates().items()}
        print(
            f"[LOG] VisionSTDP Step {step}, stage1_rate={stage1_rate:.4f}, "
            f"stage2_rate={stage2_rate:.4f}, stage3_rate={stage3_rate:.4f}, "
            f"weight_mean={weight_mean:.4f}, stdp_update_norm={update_norm:.6f}"
        )
        print(f"[LOG] Gates: {gates_info}")


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model = VisionMultiStageSNN(
        input_size=32 * 32 * 3,
        stage1_size=args.stage1_size,
        stage2_size=args.stage2_size,
        stage3_size=args.stage3_size,
    ).to(device)
    dataloader = prepare_dataloader(args)

    for epoch in range(1, args.epochs + 1):
        print(f"=== Epoch {epoch}/{args.epochs} ===")
        train_epoch(model, dataloader, device, args)


if __name__ == "__main__":
    main()
