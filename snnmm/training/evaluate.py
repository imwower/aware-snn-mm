import argparse
import torch
from torch.utils.data import DataLoader

from snnmm.datasets.cifar100 import CIFAR100Dataset
from snnmm.encoding.text_encoding import label_poisson_encode
from snnmm.encoding.vision_encoding import poisson_encode
from snnmm.models.core import AwarenessCoreSNN
from snnmm.models.text_path import LabelSNN
from snnmm.models.vision_path import VisionMultiStageSNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate R-STDP trained model on CIFAR-100.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--data-root", type=str, default="data/cifar-100-python")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--limit-steps", type=int, default=None)
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
    dataset = CIFAR100Dataset(root=args.data_root, train=False)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)

    vision = VisionMultiStageSNN().to(device)
    text_model = LabelSNN().to(device)
    core = AwarenessCoreSNN(vis_dim=vision.stage3[0].out_features, text_dim=text_model.fc2.out_features).to(device)

    if "vision" in ckpt:
        vision.load_state_dict(ckpt["vision"])
    if "text_model" in ckpt:
        text_model.load_state_dict(ckpt["text_model"])
    if "core" in ckpt:
        core.load_state_dict(ckpt["core"])
    if "gates" in ckpt:
        for k, v in ckpt["gates"].items():
            vision.gates[k].data.copy_(v.to(device))

    dataloader = prepare_dataloader(args)
    total = 0
    correct = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader, start=1):
            if args.limit_steps is not None and step > args.limit_steps:
                break
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            total += labels.numel()

            vis_spikes = poisson_encode(images, timesteps=args.timesteps, max_rate=1.0, flatten=True)
            text_spikes = label_poisson_encode(labels, timesteps=args.timesteps, n_labels=100, device=device)

            _, _, h_high = vision(vis_spikes)
            _, h_text_sem = text_model(text_spikes)
            _, _, _, cls_spikes = core(h_high, h_text_sem, timesteps=args.timesteps, cycle_length=10)

            rates = cls_spikes.mean(dim=0)
            pred = rates.argmax(dim=1)
            correct += (pred == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    print(f"[LOG] Eval accuracy: {acc:.4f} over {total} samples")


if __name__ == "__main__":
    main()
