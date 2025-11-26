import os
import pickle
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class CIFAR100Dataset(Dataset):
    """
    Minimal CIFAR-100 loader using the official python format.

    Returns (image_tensor, fine_label_int, fine_label_name).
    Image tensor shape: (3, 32, 32), float32 in [0, 1].
    """

    def __init__(
        self,
        root: str = "data/cifar-100-python",
        train: bool = True,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        split = "train" if train else "test"
        self.data_path = os.path.join(root, split)
        self.meta_path = os.path.join(root, "meta")
        self.transform = transform

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"CIFAR-100 split not found at {self.data_path}. "
                "Download the python version and extract to data/cifar-100-python/."
            )
        self.images, self.labels = self._load_split(self.data_path)
        self.label_names = self._load_meta(self.meta_path)

    @staticmethod
    def _load_split(path: str) -> Tuple[torch.Tensor, List[int]]:
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        data = entry["data"]  # shape (N, 3072)
        images = torch.tensor(data, dtype=torch.float32).view(-1, 3, 32, 32) / 255.0
        labels: List[int] = entry["fine_labels"]
        return images, labels

    @staticmethod
    def _load_meta(path: str) -> List[str]:
        with open(path, "rb") as f:
            entry = pickle.load(f, encoding="latin1")
        return entry["fine_label_names"]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        label_name = self.label_names[label]
        return img, label, label_name

