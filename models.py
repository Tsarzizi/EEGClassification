import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, _ = self.lstm(x, (h0, c0))
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output


if __name__ == "__main__":
    # 定义输入维度、隐藏层维度、LSTM层数和输出维度
    input_size = 48000
    hidden_size = 200
    num_layers = 2
    output_size = 3

    # 创建LSTM模型
    lstm = Net(input_size, output_size, hidden_size, num_layers)

    # 随机生成输入数据
    input_data = torch.randn(12, 62, 48000)
    print(input_data.shape)

    # 前向传播
    output = lstm(input_data)

    print(type(output[0]))

