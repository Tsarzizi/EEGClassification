import pickle
import time
from socket import socket

import acl
import acllite_utils as utils
from acllite_model import AclLiteModel
from acllite_resource import resource_list


class AclLiteResource:
    """
    AclLiteResource类
    """

    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
        self.run_mode = None

    def init(self):
        """
        初始化资源
        """
        print("init resource stage:")
        ret = acl.init()  # acl初始化

        ret = acl.rt.set_device(self.device_id)  # 指定运算的device
        utils.check_ret("acl.rt.set_device", ret)

        self.context, ret = acl.rt.create_context(self.device_id)  # 创建context
        utils.check_ret("acl.rt.create_context", ret)

        self.stream, ret = acl.rt.create_stream()  # 创建stream
        utils.check_ret("acl.rt.create_stream", ret)

        self.run_mode, ret = acl.rt.get_run_mode()  # 获取运行模式
        utils.check_ret("acl.rt.get_run_mode", ret)

        print("Init resource success")

    def __del__(self):
        print("acl resource release all resource")
        resource_list.destroy()
        if self.stream:
            print("acl resource release stream")
            acl.rt.destroy_stream(self.stream)  # 销毁stream

        if self.context:
            print("acl resource release context")
            acl.rt.destroy_context(self.context)  # 释放context

        print("Reset acl device ", self.device_id)
        acl.rt.reset_device(self.device_id)  # 释放device

        print("Release acl resource success")


def om_predict(data):
    acl_resource = AclLiteResource()
    acl_resource.init()
    # 加载模型
    model = AclLiteModel('./checkpoints/eeg.om')
    # 计算推理耗时
    start_time = time.time()
    # 执行推理
    result = model.execute([data])
    end_time = time.time()
    print('time:{}ms'.format(end_time - start_time))
    return result[0].argmax(1)


def predict():
    # 创建套接字
    tcp_server = socket()
    # 绑定ip，port
    # 这里ip默认本机
    address = ('', 8000)
    tcp_server.bind(address)
    # 启动被动连接
    # 多少个客户端可以连接
    tcp_server.listen(128)
    # 使用socket创建的套接字默认的属性是主动的
    # 使用listen将其变为被动的，这样就可以接收别人的链接了

    # 创建接收
    # 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
    while True:
        client_socket, clientAddr = tcp_server.accept()
        # client_socket用来为这个客户端服务，相当于的tcp_server套接字的代理
        # tcp_server_socket就可以省下来专门等待其他新客户端的链接
        # 这里clientAddr存放的就是连接服务器的客户端地址
        # 接收对方发送过来的数据
        data_len = client_socket.recv(6)
        data_len = int(data_len.decode('gbk', 'ignore'))
        data = b""
        while True:
            receive_data = client_socket.recv(1024)
            data += receive_data
            print(len(data), '/', data_len)
            if len(data) == data_len:
                break  # 接收1024给字节,这里recv接收的不再是元组，区别UDP
        data = pickle.loads(data)
        print("已接收数据", data.shape, type(data))
        pred = str(int(om_predict(data)))
        # 发送数据给客户端
        client_socket.send(pred.encode())

        client_socket.close()
