from collections.abc import Iterable

from loguru import logger
from torch import nn
from torch.optim import SGD, Adam, Optimizer


# TODO: 补充更多内置优化器
OPTIMIZERS = {"SGD": SGD, "ADAM": Adam}


class OptimizerBuilder:
    def __init__(self, optimizer_name: str = "SGD") -> None:
        """
        optimizer_name:
            - SGD
            - Adam
        """
        self.optimizer_name = optimizer_name.upper()

    def build(self, params: Iterable[nn.Parameter], **kwargs: float) -> Optimizer:
        """params 来自 net.parameters()，所以必须先有 net 才能建 optimizer。"""
        if self.optimizer_name not in OPTIMIZERS:
            raise ValueError(
                f"unsupported optimizer: {self.optimizer_name!r}, "
                f"available: {sorted(OPTIMIZERS)}"
            )

        logger.info(f"Using {self.optimizer_name} {kwargs}")
        return OPTIMIZERS[self.optimizer_name](params, **kwargs)
