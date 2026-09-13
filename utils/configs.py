from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch

# torch 支持 "cpu" / "cuda" / "cuda:0" / "mps" 等字符串，也支持 torch.device 对象
Device: TypeAlias = str | torch.device


@dataclass
class Config:
    """一次训练的全部可变项：用哪个数据集、哪个网络、怎么训。

    形状信息（in_dim / out_dim）不在这里——它由数据集决定，放在
    utils/datasets.py 的 DatasetSpec 里，网络自己去取。
    """

    dataset: str = "MNIST"
    net: str = "MLP"
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    device: Device = "cpu"
    # 网络自己的超参，key 必须是该网络 __init__ 的参数名，builder 会校验
    net_kwargs: dict[str, Any] = field(default_factory=dict)

    @property
    def run_name(self) -> str:
        """结果目录名，比如 mnist_mlp。"""
        return f"{self.dataset.lower()}_{self.net.lower()}"
