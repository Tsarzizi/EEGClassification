import random

from torch.optim import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import *
from mydataset import EEG_Dataset
from utils import *

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

PATH = 'Z:/Data.zip'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
SHUFFLE = True

dataset = EEG_Dataset(PATH)
train_dataloader, val_dataloader = get_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)

INPUT_SIZE = 48000
OUTPUT_SIZE = 3
HIDDEN_SIZE = 200
NUM_LAYERS = 2


net = Net(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = Adam(net.parameters())

scheduler = ReduceLROnPlateau(optimizer)


def train(epoch, net, optimizer, criterion, train_dataloader):
    net.train()
    for i in range(epoch):
        num_examples = 0
        total_loss = 0
        total_correct = 0
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data = data.to(DEVICE)
            target = target.to(DEVICE).float()
            '''获取模型输出并计算损失'''
            output = net(data).float()
            loss = criterion(output, target)

            '''1.清空梯度 2.反向传播求梯度 3.优化参数'''
            optimizer.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()

            '''计算当前epoch中：1.预测正确的样本总数 2.总损失'''
            num_examples += len(target)
            batch_preds = output.argmax(dim=-1)
            correct = (batch_preds == target).sum().item()
            total_correct += correct
            total_loss += loss.item()

            '''计算准确率和平均损失，并显示在tqdm中'''
            accuracy = total_correct / num_examples
            avg_loss = total_loss / num_examples
            print(f'accuracy:{accuracy} , avg_loss:{avg_loss}')
            torch.save(net, 'model.pth')
if __name__ == '__main__':
    train(3, net, optimizer, criterion, train_dataloader)
