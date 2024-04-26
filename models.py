import torch
import torch.nn as nn


class EEGNetBlock(nn.Module):
    def __init__(self):
        super(EEGNetBlock, self).__init__()
        self.F1 = 16
        self.F2 = 16
        self.D = 2
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, self.F1, (1, 100)),
            nn.ZeroPad2d((0, 63, 0, 0)),
            nn.Conv2d(self.F1, self.F1 * self.D, (62, 1), groups=self.F1),
            nn.BatchNorm2d(self.F1 * self.D, False),
            nn.ELU(inplace=True),
            nn.AvgPool2d(1, 4),
            nn.Dropout2d(0.25)
        )

        # self.conv1 = nn.Conv2d(1, self.F1, (1, 100))
        # self.padding1 = nn.ZeroPad2d((0, 63, 0, 0))
        # self.depthwise1 = nn.Conv2d(self.F1, self.F1 * self.D, (62, 1), groups=self.F1)
        # self.batchnorm1 = nn.BatchNorm2d(self.F1 * self.D, False)
        # self.elu = nn.ELU(inplace=True)
        # self.pooling1 = nn.AvgPool2d(1, 4)
        # self.dropout = nn.Dropout2d(0.25)

        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D * self.D, (1, 16), groups=self.F1 * self.D),
            nn.ZeroPad2d((0, 15, 0, 0)),
            nn.Conv2d(self.F1 * self.D * self.D, self.F2, 1),
            nn.BatchNorm2d(self.F2, False),
            nn.AvgPool2d(1, 4)
        )
        # self.depthwise2 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D * self.D, (1, 16), groups=self.F1 * self.D)
        # self.padding2 = nn.ZeroPad2d((0, 15, 0, 0))
        # self.pointwise = nn.Conv2d(self.F1 * self.D * self.D, self.F2, 1)
        # self.batchnorm2 = nn.BatchNorm2d(self.F2, False)
        # self.pooling2 = nn.AvgPool2d(1, 4)

        # 全连接层
        # 此维度将取决于数据中每个样本的时间戳数。
        # I have 120 timepoints.
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(self.F2 * self.T // (4 * 4), 3)
        self.fc1 = nn.Linear(240, 3)

    def forward(self, x):
        # input [1, 1, 62, 48000]
        # x = x.view(x.shape[0], 5, 62, 265)

        x = self.layer1(x)
        # x = self.conv1(x)  # [1, 64, 62, 47937]
        # x = self.padding1(x)  # [1, 64, 62, 48000]
        # x = self.depthwise1(x)  # [1, 64 * 2, 1, 48000]
        # x = self.batchnorm1(x)
        # x = self.elu(x)
        # x = self.pooling1(x)  # [1, 64 * 2, 1, 12000]
        # x = self.dropout(x)
        #
        x = self.layer2(x)
        # x = self.depthwise2(x)  # [1, 64 * 2 * 2, 1, 11985]
        # x = self.padding2(x)  # [1, 64 * 2 * 2, 1, 12000]
        # x = self.pointwise(x)  # [1, 64 * 2 * 2, 1, 12000]
        # x = self.batchnorm2(x)  # [1, 64 * 2 * 2, 1, 12000]
        # x = self.elu(x)  # [1, 64 * 2 * 2, 1, 12000]
        # x = self.pooling2(x)  # [1, 64 * 2 * 2, 1, 1500]
        # x = self.dropout(x)  # [1, 64 * 2 * 2, 1, 1500]
        # #
        # # # Layer 3
        # # 全连接层
        x = self.flatten(x)  # [1, 64 * 2 * 2 * 1500]
        # x = x.reshape(-1, 1, self.F2 * 625)
        x = self.fc1(x)  # [1, 3]
        # x = torch.sigmoid(x)  # [1, 3]

        return x


if __name__ == "__main__":
    # 定义输入维度、隐藏层维度、LSTM层数和输出维度

    # 随机生成输入数据
    input_data = torch.randn(10, 5, 62, 265)

    # model = TemporalConvNet(310, [64, 32, 16, 8])
    model = EEGNetBlock()
    # print(input_data.shape)

    # 前向传播
    output = model(input_data)
    # output = eegnet(input_data)
    print(output.shape)
    print(output)
