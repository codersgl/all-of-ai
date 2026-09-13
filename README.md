all-of-ai 是一个可插拔的图像分类训练项目：数据集、网络、损失函数、优化器都注册在各自的表里，用 `uv run python train/train.py --dataset KMNIST --net ALEXNET` 这样的命令就能换数据集或换网络跑训练，训练历史自动落盘到 `results/`。
