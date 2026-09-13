"""分类训练入口：数据集和网络都是配置项，切换靠 --dataset / --net，不用改代码。

     uv run python train/train.py --dataset KMNIST --net ALEXNET --net-arg dropout=0.2
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from DeepLearning.nets.builder import NETS, NetBuilder
from utils.configs import Config
from utils.criterions import CriterionBuilder
from utils.datasets import DATASETS, DatasetLoader
from utils.optimizers import OptimizerBuilder
from utils.paths import RESULTS_DIR
from utils.trainers import ClassificationTrainer, TrainerBuilder


def build_trainer(cfg: Config) -> ClassificationTrainer:
    """按 cfg 组装数据集、模型、损失、优化器，返回配好的 trainer。"""
    loader = DatasetLoader(cfg.dataset)
    train_set, test_set = loader.load()

    # 输入形状和类别数都取自数据集，网络和配置里都不手写
    net = NetBuilder(cfg.net).build(
        loader.spec.in_shape, loader.spec.num_classes, **cfg.net_kwargs
    )
    criterion = CriterionBuilder("CrossEntropyLoss").build()
    # optimizer 要 net.parameters()，所以必须在 net 之后构建
    optimizer = OptimizerBuilder("Adam").build(net.parameters(), lr=cfg.lr)

    return TrainerBuilder("CLASSIFICATION").build(
        net=net,
        train_dataset=train_set,
        test_dataset=test_set,
        criterion=criterion,
        optimizer=optimizer,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        device=cfg.device,
    )


def _parse_net_arg(text: str) -> tuple[str, Any]:
    """把 --net-arg hid_dim=256 拆成 ("hid_dim", 256)。"""
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"--net-arg 要 key=value 形式，收到 {text!r}")

    lowered = value.lower()
    if lowered in {"none", "null"}:
        return key, None
    if lowered in {"true", "false"}:
        return key, lowered == "true"
    for cast in (int, float):
        try:
            return key, cast(value)
        except ValueError:
            continue
    return key, value


def parse_args(argv: list[str] | None = None) -> tuple[Config, Path | None]:
    defaults = Config()
    parser = argparse.ArgumentParser(description="分类训练，数据集和网络可换")
    parser.add_argument(
        "--dataset",
        type=str.upper,
        default=defaults.dataset,
        choices=sorted(DATASETS),
        help="用哪个数据集",
    )
    parser.add_argument(
        "--net",
        type=str.upper,
        default=defaults.net,
        choices=sorted(NETS),
        help="用哪个网络",
    )
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument(
        "--net-arg",
        action="append",
        default=[],
        metavar="K=V",
        help="传给网络的超参，可重复，例如 --net-arg hid_dim=256 --net-arg dropout=0.2",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="结果目录，默认 results/<dataset>_<net>",
    )
    args = parser.parse_args(argv)

    cfg = Config(
        dataset=args.dataset,
        net=args.net,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        net_kwargs=dict(_parse_net_arg(text) for text in args.net_arg),
    )
    return cfg, args.out


def save_history(cfg: Config, history: list[dict[str, float]], out_dir: Path) -> Path:
    """把 config 和 history 写成带时间戳的 json，返回实际写出的路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    payload = {
        "config": {
            k: str(v) for k, v in asdict(cfg).items()
        },  # device 可能是 torch.device
        "history": history,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"history 已写入 {path}")
    return path


def main(
    cfg: Config | None = None, out_dir: Path | None = None
) -> list[dict[str, float]]:
    cfg = cfg or Config()
    out_dir = out_dir or RESULTS_DIR / cfg.run_name
    logger.info(f"{cfg}")

    history = build_trainer(cfg).train()
    if history:
        logger.info(f"best test acc: {max(r['test_acc'] for r in history):.4f}")
    save_history(cfg, history, out_dir)
    return history


if __name__ == "__main__":
    main(*parse_args())
