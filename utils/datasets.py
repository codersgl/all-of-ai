from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from loguru import logger
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from utils.paths import DATASETS_DIR

# 下到仓库的 DataSets/ 里
DEFAULT_ROOT = DATASETS_DIR


@dataclass(frozen=True)
class DatasetSpec:
    """一个数据集的全部元信息，网络和配置都从这里取数，不再手写 784 / 10 这种数字。"""

    factory: Callable[..., Dataset]
    in_shape: tuple[int, int, int]  # (c, h, w)
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    transform: Callable[[Any], Tensor]

    @property
    def num_channels(self) -> int:
        return self.in_shape[0]


def _spec(
    factory: Callable[..., Dataset],
    in_shape: tuple[int, int, int],
    num_classes: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> DatasetSpec:
    """归一化统计量跟数据集绑定，所以 transform 在这里就地拼好。"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    return DatasetSpec(factory, in_shape, num_classes, mean, std, transform)


# 数据集名 -> 元信息。新增数据集 = 在这里加一行。
# EMNIST 是 byclass，62 类：类别数由这里提供，配置里不再手写 out_dim。
DATASETS = {
    "MNIST": _spec(datasets.MNIST, (1, 28, 28), 10, (0.1307,), (0.3081,)),
    # TODO: KMNIST / EMNIST 暂时沿用 MNIST 的统计量，想更准就自己算一遍训练集
    "KMNIST": _spec(datasets.KMNIST, (1, 28, 28), 10, (0.1307,), (0.3081,)),
    "FashionMNIST": _spec(
        datasets.FashionMNIST, (1, 28, 28), 10, (0.2860,), (0.3530,)
    ),
    "EMNIST": _spec(
        partial(datasets.EMNIST, split="byclass"),
        (1, 28, 28),
        62,
        (0.1307,),
        (0.3081,),
    ),
}


class DatasetLoader:
    def __init__(
        self,
        dataset_name: str = "MNIST",
        root: Path = DEFAULT_ROOT,
        transform: Callable[[Any], Tensor] | None = None,
    ) -> None:
        if dataset_name not in DATASETS:
            raise ValueError(
                f"unknown dataset: {dataset_name!r}, available: {sorted(DATASETS)}"
            )

        self.dataset_name = dataset_name
        self.spec = DATASETS[dataset_name]
        self.root = root
        # 默认用数据集自己的 transform，要加增广再传进来
        self.transform = transform or self.spec.transform

    def load(self) -> tuple[Dataset, Dataset]:
        factory = self.spec.factory
        train_set = factory(
            root=self.root, train=True, download=True, transform=self.transform
        )
        test_set = factory(
            root=self.root, train=False, download=True, transform=self.transform
        )

        logger.info(
            f"{self.dataset_name}: train set size {len(train_set)}, "
            f"test set size: {len(test_set)}, sample shape: {train_set[0][0].shape}"
        )
        return train_set, test_set


if __name__ == "__main__":
    loader = DatasetLoader(dataset_name="MNIST")
    train_set, test_set = loader.load()
