import numpy as np
import torch
from torch.utils.data import random_split, DataLoader

from mydataset import EEG_Dataset


def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    temp = np.zeros((len(batch), 62, 200 * 240))
    for i in range(len(batch)):
        print(f'数据加载中{i}/{len(batch)}')
        if data[i].shape[1] > 200 * 240:
            width = 200 * 240
            temp[i][:, :width] = data[i][:, :width]
        else:
            temp[i][:, :data[i].shape[1]] = data[i]
        data[i] = temp[i]
    data = np.array(data)
    target = np.array(target)
    data = torch.from_numpy(data).float()
    target = torch.from_numpy(target)
    print('加载完成')
    return data, target


def get_dataloader(dataset, batch_size, shuffle=True, collate_fn=collate_fn):
    EEG_dataset = dataset
    train_size = int(0.8 * len(EEG_dataset))
    val_size = len(EEG_dataset) - train_size
    train_dataset, val_dataset = random_split(EEG_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    return train_dataloader, val_dataloader


if __name__ == '__main__':
    # ZIP文件的路径
    path = 'Z:/Data.zip'
    eeg = EEG_Dataset(path)
    t, v = get_dataloader(eeg, 1, True, collate_fn)
    for d, l in t:
        print(d, l)
