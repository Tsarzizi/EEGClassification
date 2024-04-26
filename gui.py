import pickle
import time
from socket import socket

import gradio as gr
import mne
import pandas as pd
from matplotlib import pyplot as plt
import os
import torch
from utils import pre_processes
from utils import label2img
import numpy as np

ckpt_list = ['checkpoints/' + e for e in os.listdir('checkpoints')]
channels = pd.read_excel("./channel-order.xlsx", header=None)
channels = channels.values.squeeze().tolist()


def channel2index(channel):
    return channels.index(channel)


def load_fn(file):
    global data_
    path = file.name
    data_ = np.load(path)
    data1020 = pd.read_csv('./1020.csv', index_col=0)
    channels1020 = np.array(data1020.index)
    value1020 = np.array(data1020)
    list_dic = dict(zip(channels1020, value1020))
    montage_1020 = mne.channels.make_dig_montage(ch_pos=list_dic,
                                                 nasion=[5.27205792e-18, 8.60992398e-02, -4.01487349e-02],
                                                 lpa=[-0.08609924, -0., -0.04014873],
                                                 rpa=[0.08609924, 0., -0.04014873])
    info = mne.create_info(ch_names=channels1020.tolist(), ch_types=['eeg'] * len(channels1020), sfreq=200)
    raw = mne.io.RawArray(data_, info)
    raw.set_montage(montage_1020)
    return raw.info


def draw_fn(channel, time=0):
    global data_
    index = channel2index(channel)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    data_draw = data_[index]
    ax.set_title(str(channel)+' - '+str(time)+'s')
    ax.set_xlabel('Time')
    ax.set_ylim(np.min(data_draw), np.max(data_draw))
    ax.plot(data_draw[time * 200:(time + 1) * 200])
    return fig


def pred_fn():
    global data_
    data = pre_processes(data_)
    data = np.expand_dims(data, 0)
    with socket() as tcp_socket:
        server_address = '192.168.0.104'
        server_port = 8000
        tcp_socket.connect((server_address, server_port))
        # 准备需要传送的数据

        send_data = pickle.dumps(data)
        data_len = len(send_data)
        tcp_socket.send(str(data_len).encode('gbk'))
        tcp_socket.send(send_data)
        # 从服务器接收数据
        # 注意这个1024byte，大小根据需求自己设置
        pred = tcp_socket.recv(64)
        pred = int(pred.decode('gbk'))
        text, img = label2img(pred)
    return text, img


if __name__ == '__main__':
    data_ = None
    with gr.Blocks() as demo:
        # demo
        with gr.Tab(label='demo'):
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        data = gr.File(label='data_load')
                        output_info = gr.Text(label='data info')
                    load_data = gr.Button("上传")
                    with gr.Group():
                        chose_channel = gr.Dropdown(label='channel_chose', choices=channels)
                        n_times = gr.Slider(label='time_select', minimum=0, maximum=250)
                with gr.Column():
                    img = gr.Plot(label='EEG image')
                    show_img = gr.Button("展示")

            with gr.Row():
                with gr.Column():

                    output_label = gr.Label()
                output_img = gr.Image()
            with gr.Column():
                pred = gr.Button('推理', variant='primary')

        load_data.click(fn=load_fn, inputs=[data], outputs=[output_info])
        show_img.click(fn=draw_fn, inputs=[chose_channel, n_times], outputs=[img])
        pred.click(fn=pred_fn, inputs=[], outputs=[output_label, output_img])

    demo.launch()
