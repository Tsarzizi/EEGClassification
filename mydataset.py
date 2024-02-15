import os
import os.path as osp
import re

import numpy as np
from torch.utils.data import Dataset

import scipy.io as sio

class EEG_Dataset(Dataset):
    def __init__(self, path):
        self.path = path
        data_list = []
        labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        data_files_name = os.listdir(self.path)

        for f in data_files_name:
            match = re.match('^\d.*mat$', f)
            if match is not None:
                f = match.group()
                file_name = osp.join(self.path, f)
                print('Loading', file_name)

                datas = sio.loadmat(file_name)
                keys = list(datas.keys())
                for i in range(3, len(keys)):
                    data_map = {'data': datas[keys[i]], 'label': labels[i - 3]}
                    data_list.append(data_map)
        self.data_list = data_list

        # f = 'D:\Data\EEG\SEED\Preprocessed_EEG\\10_20131130.mat'
        # datas = sio.loadmat(f)
        # keys = list(datas.keys())
        # for i in range(3, len(keys)):
        #     data_map = {'data': datas[keys[i]], 'label': labels[i - 3]}
        #     data_list.append(data_map)
        # self.data_list = data_list

    def __getitem__(self, idx):
        self.data = np.array(self.data_list[idx]['data'])
        self.label = np.array(self.data_list[idx]['label']) + 1

        return self.data, self.label

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    path = 'D:\Data\EEG\SEED\\Preprocessed_EEG\\'
    eeg = EEG_Dataset(path)
    print(eeg)