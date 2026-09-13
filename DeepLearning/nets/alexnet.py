from torch import Tensor, nn


class AlexNet(nn.Module):
    """AlexNet 的结构：5 个卷积 + 3 个全连接，中间穿插 MaxPool 和 Dropout。

    相对原版（224x224 输入）只改了两处，都是为了能吃小图：

    - 首层从 11x11/stride 4 改成 5x5/stride 1。原版在 28x28 上过完第一层只剩 5x5，
      最后一个 3x3 MaxPool 会因为输出尺寸算成负数直接报错；
    - 分类头之前加 AdaptiveAvgPool2d(6)，把特征图固定成 6x6，于是 28x28
      （MNIST 系）和 32x32（CIFAR 系）都能直接吃，换数据集不用改网络。

    注意 28x28 的图经过三次池化后特征图只剩 2x2，这个网络更擅长 >=32x32 的输入。
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        num_classes: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        channels = in_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(6)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.pool(self.features(x)))
