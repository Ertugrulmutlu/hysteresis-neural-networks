# src/data.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def _build_transform(config):
    tfms = [transforms.ToTensor()]
    if config["data"]["normalize"]:
        tfms.append(
            transforms.Normalize(
                mean=config["data"]["normalize_mean"],
                std=config["data"]["normalize_std"]
            )
        )
    return transforms.Compose(tfms)

def _indices_by_digits(dataset, digits_set):
    # dataset[i] -> (x, y)
    return [i for i, (_, y) in enumerate(dataset) if y in digits_set]

def get_mnist_datasets(config):
    transform = _build_transform(config)

    train = datasets.MNIST(
        root=config["data"]["root"],
        train=True,
        download=True,
        transform=transform
    )
    test = datasets.MNIST(
        root=config["data"]["root"],
        train=False,
        download=True,
        transform=transform
    )

    A_digits = set(config["split"]["A_digits"])
    B_digits = set(config["split"]["B_digits"])

    A_train_idx = _indices_by_digits(train, A_digits)
    B_train_idx = _indices_by_digits(train, B_digits)

    A_test_idx  = _indices_by_digits(test, A_digits)
    B_test_idx  = _indices_by_digits(test, B_digits)
    full_test_idx = list(range(len(test)))

    return train, test, A_train_idx, B_train_idx, A_test_idx, B_test_idx, full_test_idx

def make_loader(dataset, indices, config, seed, shuffle):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=config["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["data"]["num_workers"],
        generator=g
    )
