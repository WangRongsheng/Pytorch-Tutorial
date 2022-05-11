from torch import nn

# 创建网络模型
# 结构实现参考：./CIFmodel.png
class CIFmodel(nn.Module):
    def __init__(self):
        super(CIFmodel, self).__init__()
        self.model = nn.Sequential(
            # 输入3通道，输出32通道，kernel_size=5，stride=1，计算得出padding=2
            nn.Conv2d(3, 32, 5, 1, 2),
            # kernel_size=2
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            # 拉平
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x