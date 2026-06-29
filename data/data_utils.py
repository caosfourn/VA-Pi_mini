"""
Data loading cho VA-π Mini.

- CIFAR-10: dùng để TRAIN FROM SCRATCH (Implement: Mini).
- CIFAR-100, STL-10: dùng làm 2 "novel dataset" ở giai đoạn Evaluation
  (VA-π RL fine-tuning) — model KHÔNG được train từ đầu trên các dataset này.

Ảnh được chuẩn hoá về [-1, 1] để khớp với Tanh() ở output của Decoder.
"""

import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


def _to_pm1(x):
    return x * 2.0 - 1.0  # [0,1] -> [-1,1]


def get_transform(image_size: int):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Lambda(_to_pm1),
    ])


def get_cifar10_loaders(root="./data", image_size=32, batch_size=64,
                         num_workers=2, train_subset=None):
    """Dataset Mini cho train-from-scratch. 10 class -> num_classes=10."""
    transform = get_transform(image_size)
    train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True,
                                              transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True,
                                             transform=transform)
    if train_subset is not None:
        train_set = Subset(train_set, list(range(min(train_subset, len(train_set)))))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10  # num_classes


def get_cifar100_loaders(root="./data", image_size=32, batch_size=64,
                          num_workers=2, subset=None):
    """Novel dataset #1 cho Evaluation. 100 class."""
    transform = get_transform(image_size)
    test_set = torchvision.datasets.CIFAR100(root=root, train=False, download=True,
                                              transform=transform)
    if subset is not None:
        test_set = Subset(test_set, list(range(min(subset, len(test_set)))))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return loader, 100


def get_stl10_loaders(root="./data", image_size=32, batch_size=64,
                       num_workers=2, subset=None):
    """Novel dataset #2 cho Evaluation. STL-10 'train' split có 10 class có nhãn.
    Resize về cùng image_size với CIFAR để dùng được cùng VQVAE/GPT đã train."""
    transform = get_transform(image_size)
    test_set = torchvision.datasets.STL10(root=root, split="test", download=True,
                                           transform=transform)
    if subset is not None:
        test_set = Subset(test_set, list(range(min(subset, len(test_set)))))
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=True)
    return loader, 10


if __name__ == "__main__":
    # smoke test nhanh (chỉ chạy thử cấu trúc, không cần internet trong sandbox này)
    print("Module data_utils.py đã sẵn sàng.")
    print("Gọi get_cifar10_loaders(), get_cifar100_loaders(), get_stl10_loaders()")
    print("khi train trên Kaggle (cần internet để torchvision tự download).")
