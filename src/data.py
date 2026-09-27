"""MNIST dataset and deterministic loader helpers."""
from typing import Any, Callable, Generic, Iterable, Iterator, Sequence, TypeVar

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as transform_functional

Batch = TypeVar("Batch")


def build_transform(config: dict[str, Any]) -> transforms.Compose:
    items: list[Any] = [transforms.ToTensor()]
    augmentation = config["data"].get("augmentation", "none")
    if augmentation != "none":
        raise ValueError("Only augmentation='none' is supported for the Part 1 protocol")
    if config["data"].get("normalize", False):
        items.append(transforms.Normalize(config["data"]["normalize_mean"], config["data"]["normalize_std"]))
    return transforms.Compose(items)


def indices_by_digits(dataset: Dataset, digits: Iterable[int]) -> list[int]:
    wanted = set(digits)
    targets = getattr(dataset, "targets", None)
    if targets is not None:
        return [i for i, y in enumerate(targets) if int(y) in wanted]
    return [i for i, (_, y) in enumerate(dataset) if int(y) in wanted]


def get_mnist_datasets(config: dict[str, Any], download: bool = True):
    protocol = validate_data_protocol(config["data"])
    if protocol == "rotated_mnist":
        return get_rotated_mnist_datasets(config, download)
    transform = build_transform(config)
    root = config["data"]["root"]
    train = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    test = datasets.MNIST(root=root, train=False, download=download, transform=transform)
    a, b = config["split"]["A_digits"], config["split"]["B_digits"]
    return (train, test, indices_by_digits(train, a), indices_by_digits(train, b),
            indices_by_digits(test, a), indices_by_digits(test, b), list(range(len(test))))


def validate_data_protocol(data_config: dict[str, Any]) -> str:
    protocol = str(data_config.get("protocol", "class_split"))
    if protocol not in {"class_split", "rotated_mnist"}:
        raise ValueError("data.protocol must be class_split or rotated_mnist")
    if protocol == "rotated_mnist":
        for key in ("rotation_degrees_A", "rotation_degrees_B"):
            value = data_config.get(key)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a numeric fixed angle")
    return protocol


class FixedRotationDataset(Dataset):
    """Deterministic fixed-angle view of an underlying image dataset."""

    def __init__(self, base: Dataset, angle: float, transform: Callable[[Any], Any]) -> None:
        self.base, self.angle, self.transform = base, float(angle), transform
        self.targets = getattr(base, "targets", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label = self.base[index]
        return self.transform(transform_functional.rotate(image, self.angle)), label


def get_rotated_mnist_datasets(config: dict[str, Any], download: bool = True):
    data = config["data"]; root = data["root"]; transform = build_transform(config)
    train_base = datasets.MNIST(root=root, train=True, download=download, transform=None)
    test_base = datasets.MNIST(root=root, train=False, download=download, transform=None)
    train_domains = [FixedRotationDataset(train_base, data["rotation_degrees_A"], transform),
                     FixedRotationDataset(train_base, data["rotation_degrees_B"], transform)]
    test_domains = [FixedRotationDataset(test_base, data["rotation_degrees_A"], transform),
                    FixedRotationDataset(test_base, data["rotation_degrees_B"], transform)]
    train, test = ConcatDataset(train_domains), ConcatDataset(test_domains)
    n_train, n_test = len(train_base), len(test_base)
    return train, test, list(range(n_train)), list(range(n_train, 2*n_train)), list(range(n_test)), list(range(n_test, 2*n_test)), list(range(2*n_test))


def make_loader(dataset: Dataset, indices: Sequence[int], config: dict[str, Any], seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, list(indices)), batch_size=int(config["data"]["batch_size"]),
                      shuffle=shuffle, num_workers=int(config["data"]["num_workers"]), generator=generator)


def balanced_indices(dataset: Dataset, digits: Sequence[int], samples_per_class: int) -> list[int]:
    """Select the first equal-sized deterministic subset for every requested class."""
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than 0")
    selected: list[int] = []
    for digit in sorted(digits):
        candidates = indices_by_digits(dataset, [digit])
        if len(candidates) < samples_per_class:
            raise ValueError(f"Digit {digit} has only {len(candidates)} examples")
        selected.extend(candidates[:samples_per_class])
    return selected


class RestartableLoaderIterator(Generic[Batch]):
    """Repeat a loader factory, restarting its deterministic order each pass."""

    def __init__(self, loader_factory: Callable[[], Iterable[Batch]]) -> None:
        self.loader_factory = loader_factory
        self._iterator: Iterator[Batch] = iter(loader_factory())

    def __iter__(self) -> "RestartableLoaderIterator[Batch]":
        return self

    def __next__(self) -> Batch:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader_factory())
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise ValueError("Cannot restart an empty loader") from exc


def make_balanced_restartable_loader(
    dataset: Dataset, config: dict[str, Any], seed: int, samples_per_class: int
) -> RestartableLoaderIterator:
    """Build an infinite balanced 0-9 iterator with reproducible shuffled batches."""
    indices = balanced_indices(dataset, list(range(10)), samples_per_class)

    def factory() -> DataLoader:
        return make_loader(dataset, indices, config, seed, shuffle=True)

    return RestartableLoaderIterator(factory)


def make_relaxation_loader(dataset: Dataset, config: dict[str, Any], seed: int, samples_per_class: int):
    """Build the protocol-specific balanced C iterator."""
    if validate_data_protocol(config["data"]) == "class_split":
        return make_balanced_restartable_loader(dataset, config, seed, samples_per_class)
    selected = balanced_rotated_indices(dataset, samples_per_class)
    return RestartableLoaderIterator(lambda: make_loader(dataset, selected, config, seed, True))


def balanced_rotated_indices(dataset: Dataset, samples_per_class: int) -> list[int]:
    """Return equal digit counts from each of two deterministic rotated domains."""
    if not isinstance(dataset, ConcatDataset) or len(dataset.datasets) != 2:
        raise ValueError("rotated_mnist relaxation expects two concatenated domain datasets")
    first, second = dataset.datasets; selected = balanced_indices(first, range(10), samples_per_class)
    return selected + [len(first) + index for index in balanced_indices(second, range(10), samples_per_class)]
