import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN2(nn.Module):
    def __init__(self, n_observations, n_actions, n_frames_to_stack):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(n_frames_to_stack, 16, kernel_size=6, stride=3)
        self.conv2 = nn.Conv2d(16, 24, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(4056, 108)  # 添加一个线性层
        self.fc = nn.Linear(108, n_actions)  # 这里假设前面的池化操作后输出大小是1600（实际应根据具体情况调整）

    def forward(self, x):
        #print(x.shape)
        # print("x:",x.shape)
        x = x.float() / 255.0  # 标准化输入值
        # print("x/255:",x.shape)
        #print(x.shape)
        x = F.relu(self.conv1(x))
        # print("conv1x:",x.shape)
        # print("poolx:",x.shape)
        #print(x.shape)
        x = F.relu(self.conv2(x))
        # print("conv2x:",x.shape)
        #print(x.shape)
        #x = F.max_pool2d(x, kernel_size=2, stride=2)
        # print("poolx:",x.shape)
        # 将特征张量展平成一维向量
        # 注意：此处的1600应与定义中的全连接层输入尺寸一致
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        # print("xview:",x.shape)
        x = F.relu(self.linear(x))  # 使用新添加的线性层进行前向传播
        # print("xline:",x.shape)
        #print(x.shape)
        return self.fc(x)
