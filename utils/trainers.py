import torch
from loguru import logger
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.configs import Device


class ClassificationTrainer:
    """单标签分类的训练/评估循环，跟具体数据集和网络都无关。"""

    def __init__(
        self,
        net: nn.Module,
        train_dataset: Dataset,
        criterion: nn.Module,
        optimizer: Optimizer,
        epochs: int,
        device: Device = "cpu",
        batch_size: int = 64,
        test_dataset: Dataset | None = None,
    ) -> None:
        self.net = net.to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_loader = (
            DataLoader(test_dataset, batch_size=batch_size)
            if test_dataset is not None
            else None
        )
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

    def train(self) -> list[dict[str, float]]:
        history = []
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            total_loss, correct, seen = 0.0, 0, 0
            bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}")
            for images, labels in bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.net(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()  # 只累加数值，别把计算图带进下一个 batch
                correct += (logits.argmax(1) == labels).sum().item()
                seen += labels.size(0)
                bar.set_postfix(loss=f"{loss.item():.4f}")

            record = {
                "epoch": float(epoch),
                "train_loss": total_loss / len(self.train_loader),
                "train_acc": correct / seen,
            }
            if self.test_loader is not None:
                test_loss, test_acc = self.evaluate()
                record["test_loss"] = test_loss
                record["test_acc"] = test_acc
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"train loss {record['train_loss']:.4f} "
                    f"acc {record['train_acc']:.4f} | "
                    f"test loss {test_loss:.4f} acc {test_acc:.4f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch}/{self.epochs} | "
                    f"train loss {record['train_loss']:.4f} "
                    f"acc {record['train_acc']:.4f}"
                )
            history.append(record)

        return history

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        if self.test_loader is None:
            raise ValueError("no test_dataset given, cannot evaluate")

        self.net.eval()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits = self.net(images)
            total_loss += self.criterion(logits, labels).item()
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

        return total_loss / len(self.test_loader), correct / seen


TRAINERS = {"CLASSIFICATION": ClassificationTrainer}


class TrainerBuilder:
    def __init__(self, trainer_name: str = "CLASSIFICATION") -> None:
        self.trainer_name = trainer_name.upper()

    def build(self, **kwargs) -> ClassificationTrainer:
        if self.trainer_name not in TRAINERS:
            raise ValueError(
                f"unsupported trainer: {self.trainer_name!r}, "
                f"available: {sorted(TRAINERS)}"
            )

        logger.info(f"Using {self.trainer_name} trainer")
        return TRAINERS[self.trainer_name](**kwargs)
