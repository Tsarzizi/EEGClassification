import os
import zipfile

import numpy as np
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.labels = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
        # 打开ZIP文件
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            # 递归函数遍历ZIP文件中的所有条目
            info = zip_ref.namelist()
            # 获取数据文件名称
            self.npz = list(filter(lambda x: x.endswith('.npz'), info))

        # for f in data_files_name:
        #     match = re.match('^\d.*mat$', f)
        #     if match is not None:
        #         f = match.group()
        #         file_name = osp.join(self.path, f)
        #         print('Loading', file_name)
        #
        #         datas = sio.loadmat(file_name)
        #         keys = list(datas.keys())
        #         for i in range(3, len(keys)):
        #             data_map = {'data': datas[keys[i]], 'label': labels[i - 3]}
        #             data_list.append(data_map)
        # self.data_list = data_list

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.path, 'r') as zip_ref:
            with zip_ref.open(self.npz[idx], 'r') as f:
                self.data = np.loadtxt(f)
        self.label = self.labels[idx % len(self.labels)]

        # self.data = np.array(self.data_list[idx]['data'])
        # self.label = np.array(self.data_list[idx]['label']) + 1

        return self.data, self.label

    def __len__(self):
        return len(self.npz)


if __name__ == '__main__':
    path = 'Z:\Data.zip'
    eeg = EEGDataset(path)
    print(eeg[0][1])
