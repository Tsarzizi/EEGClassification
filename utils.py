import math

import PIL.Image
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


def pre_processes(data):
    fs = 200
    fStart = [0.5, 4, 8, 13, 32]  # 起始频率
    fEnd = [4, 8, 13, 32, 50]  # 终止频率

    preprocessed_data = []
    data = np.array(data)
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    for band_index in range(len(fStart)):
        b, a = signal.butter(4, [fStart[band_index] / fs, fEnd[band_index] / fs], 'bandpass')
        filtedData = signal.filtfilt(b, a, data)
        filtedData_de = []
        for lead in range(62):
            filtedData_split = []

            for de_index in range(0, filtedData.shape[1] - fs, fs):
                filtedData_split.append(compute_DE(filtedData[lead, de_index: de_index + fs]))

            if len(filtedData_split) < 265:
                filtedData_split += [0.5] * (265 - len(filtedData_split))
            filtedData_de.append(filtedData_split)
        filtedData_de = np.array(filtedData_de)
        preprocessed_data.append(filtedData_de)
    preprocessed_data = np.array(preprocessed_data)
    return preprocessed_data.astype(np.float32)


def label2img(label):
    if label == 0:
        return 'sad', PIL.Image.open("./label_pictures/0.png")
    elif label == 1:
        return 'neutral', PIL.Image.open("./label_pictures/1.png")
    else:
        return 'happy', PIL.Image.open("./label_pictures/2.png")
