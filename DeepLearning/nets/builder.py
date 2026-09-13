"""网络注册表 + 构建器，和 utils 里的 Criterion/Optimizer/TrainerBuilder 一个套路。

新增一个网络 = 在新文件里写个 nn.Module，然后到这里加一行。

网络约定（所有注册进来的网络都要满足）：

    __init__(self, in_shape: tuple[int, int, int], num_classes: int, **超参)
    forward(x: (b, c, h, w)) -> (b, num_classes) 的 logits

形状和类别数都由数据集提供（utils/datasets.py 的 DatasetSpec），
所以网络里不该出现 784、10 这类跟具体数据集绑死的数字。
"""

import inspect

from loguru import logger
from torch import nn

from DeepLearning.nets.alexnet import AlexNet
from DeepLearning.nets.mlp import MLPNet

NETS: dict[str, type[nn.Module]] = {
    "MLP": MLPNet,
    "ALEXNET": AlexNet,
}


class NetBuilder:
    def __init__(self, net_name: str = "MLP") -> None:
        self.net_name = net_name.upper()

    def build(
        self, in_shape: tuple[int, int, int], num_classes: int, **kwargs
    ) -> nn.Module:
        if self.net_name not in NETS:
            raise ValueError(
                f"unsupported net: {self.net_name!r}, available: {sorted(NETS)}"
            )

        net_cls = NETS[self.net_name]
        accepted = self._accepted_kwargs(net_cls)
        unknown = sorted(set(kwargs) - accepted)
        if unknown:
            raise ValueError(
                f"{self.net_name} 不认识这些参数: {unknown}, "
                f"它的可用参数: {sorted(accepted)}"
            )

        logger.info(f"Using {self.net_name} net {kwargs} on {in_shape} -> {num_classes}")
        return net_cls(in_shape, num_classes, **kwargs)

    @staticmethod
    def _accepted_kwargs(net_cls: type[nn.Module]) -> set[str]:
        """网络自己的超参名 = __init__ 参数减去固定位置的 in_shape / num_classes。"""
        params = inspect.signature(net_cls.__init__).parameters
        return set(params) - {"self", "in_shape", "num_classes"}
