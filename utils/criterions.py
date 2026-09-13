from loguru import logger
from torch import nn


# TODO: 补充更多内置损失
CRITERIONS = {"CrossEntropyLoss": nn.CrossEntropyLoss, "MSE": nn.MSELoss}


class CriterionBuilder:
    def __init__(self, criterion_name: str) -> None:
        """
        criterion_name:
            - CrossEntropyLoss
        """
        self.criterion_name = criterion_name

    def build(self, **kwargs) -> nn.Module:

        if self.criterion_name not in CRITERIONS:
            raise ValueError(
                f"unsupported criterion: {self.criterion_name!r}, "
                f"available: {sorted(CRITERIONS)}"
            )

        logger.info(f"Using {self.criterion_name} {kwargs}")
        return CRITERIONS[self.criterion_name](**kwargs)
