import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 loader using torchvision.
    Returns (image_tensor, label_int, label_name).
    Image tensor: (3,32,32) float32 in [0,1].
    """

    def __init__(self, root: str = "data", train: bool = True, download: bool = True) -> None:
        super().__init__()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dataset = datasets.CIFAR10(root=root, train=train, transform=self.transform, download=download)
        self.label_names = self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        return img, label, self.label_names[label]
