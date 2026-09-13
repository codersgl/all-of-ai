from torch import Tensor, nn


class MLPNet(nn.Module):
    """(b, c, h, w) 的图片 -> (b, num_classes) 的 logits。

    只有全连接层，所以先展平成 (b, c*h*w)；输入维度从数据集形状算出来，
    不用手写 784。约定见 DeepLearning/nets/builder.py。
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        num_classes: int,
        hid_dim: int = 128,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        channels, height, width = in_shape
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(channels * height * width, hid_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc2 = nn.Linear(hid_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(self.flatten(x)))))
