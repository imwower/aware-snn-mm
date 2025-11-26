import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.layers.gating import update_gates
from snnmm.layers.stdp import rstdp_update_linear
from snnmm.models.core import AwarenessCoreSNN
from snnmm.models.text_path import LabelSNN
from snnmm.models.vision_path import VisionMultiStageSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reward-modulated STDP training for CIFAR-100 classification.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--cycle-length", type=int, default=10)
    parser.add_argument("--lr-core", type=float, default=1e-3)
    parser.add_argument("--lr-cls", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-steps", type=int, default=10)
    parser.add_argument("--surprise-alpha", type=float, default=0.7)
    parser.add_argument("--surprise-beta", type=float, default=0.3)
    return parser.parse_args()


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


def init_traces(core: AwarenessCoreSNN, device: torch.device) -> Dict[str, torch.Tensor]:
    traces: Dict[str, torch.Tensor] = {}
    traces["pre_cls"] = torch.zeros(core.classifier.in_features, device=device)
    traces["post_cls"] = torch.zeros(core.classifier.out_features, device=device)
    traces["elig_cls"] = torch.zeros_like(core.classifier.weight, device=device)
    return traces


def compute_reward(pred: torch.Tensor, labels: torch.Tensor, surprise: torch.Tensor) -> float:
    correct = (pred == labels).float()
    base_R = torch.where(correct > 0, torch.ones_like(correct), -torch.ones_like(correct))
    # modulate by (1 - surprise)
    R = (base_R * (1.0 - surprise)).mean().item()
    return R


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
    for step, batch in enumerate(dataloader, start=1):
        if args.limit_steps is not None and step > args.limit_steps:
            break
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        total += labels.numel()

        # encode
        vis_spikes = poisson_encode(images, timesteps=args.timesteps, max_rate=1.0, flatten=True)
        text_spikes = label_poisson_encode(labels, timesteps=args.timesteps, n_labels=100, device=device)

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

        # reward and R-STDP on classifier
        R = compute_reward(pred, labels, S)
        traces["elig_cls"], traces["pre_cls"], traces["post_cls"], upd = rstdp_update_linear(
            core.classifier.weight,
            traces["elig_cls"],
            traces["pre_cls"],
            traces["post_cls"],
            pre_spikes=z_core,  # use averaged spikes as pre
            post_spikes=cls_spikes.mean(dim=0),  # averaged post spikes
            reward=R,
            lr=args.lr_cls,
        )

        # gate updates suggestion (vision stage3 as example)
        delta_g = core.suggest_gate_delta(S, num_experts=vision.num_experts)
        new_g = update_gates(vision.gates["stage3"], delta_g.to(device))
        vision.gates["stage3"].data.copy_(new_g)

        if step % 5 == 0:
            acc = correct / total if total > 0 else 0.0
            mean_S = sum(surprise_list) / len(surprise_list)
            g3 = vision.gates["stage3"].detach().cpu().numpy().tolist()
            print(
                f"[LOG] RSTDP Ep step {step}, acc={acc:.3f}, mean_S={mean_S:.3f}, g_stage3={g3[:4]}..."
            )


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    vision = VisionMultiStageSNN().to(device)
    text_model = LabelSNN().to(device)
    core = AwarenessCoreSNN(vis_dim=vision.stage3[0].out_features, text_dim=text_model.fc2.out_features).to(device)

    dataloader = prepare_dataloader(args)
    traces = init_traces(core, device)

    for epoch in range(1, args.epochs + 1):
        print(f"=== R-STDP Epoch {epoch}/{args.epochs} ===")
        train_epoch(vision, text_model, core, dataloader, device, args, traces)


if __name__ == "__main__":
    main()
