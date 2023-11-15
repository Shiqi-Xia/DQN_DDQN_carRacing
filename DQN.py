import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_frames_to_stack):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_frames_to_stack, 32, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.linear = nn.Linear(1600, 256)  # 添加一个线性层
        self.fc = nn.Linear(256, n_actions)  # 这里假设前面的池化操作后输出大小是1600（实际应根据具体情况调整）

    def forward(self, x):
        # print("x:",x.shape)
        x = x.float() / 255.0  # 标准化输入值
        # print("x/255:",x.shape)
        x = F.relu(self.conv1(x))
        # print("conv1x:",x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 池化操作可以增加非线性特征并减小尺寸
        # print("poolx:",x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2x:",x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print("poolx:",x.shape)
        # 将特征张量展平成一维向量
        # 注意：此处的1600应与定义中的全连接层输入尺寸一致
        x = x.view(x.size(0), -1)
        # print("xview:",x.shape)
        x = F.relu(self.linear(x))  # 使用新添加的线性层进行前向传播
        # print("xline:",x.shape)
        return self.fc(x)
