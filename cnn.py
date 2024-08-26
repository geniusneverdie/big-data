import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch
from torch import nn
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader
import torch

class DatasetLoader:
    def __init__(self, batch_size=32):
        # 加载数据
        with open('./data/bx.pkl', 'rb') as f:
            self.bx = np.array(pickle.load(f))
        with open('./data/bi.pkl', 'rb') as f:
            self.bi = np.array(pickle.load(f))

        # 打印bx和bi的形状
        print('Shape of bx:', self.bx.shape)
        print('Shape of bi:', self.bi.shape)


        # 其他代码不变
        with open('./data/node_idx.pkl', 'rb') as f:
            self.idx = pickle.load(f)
        with open('./data/test.pkl', 'rb') as f:
            self.test = pickle.load(f)
        with open('./data/train_user.pkl', 'rb') as f:
            self.train_user_data = pickle.load(f)

        # 检查物品ID是否在idx中，并替换不在idx中的物品ID
        for user_id, items in self.train_user_data.items():
            for i, (item_id, rating) in enumerate(items):
                if item_id not in self.idx:
                    items[i] = (-1, rating)
                else:
                    items[i] = (self.idx[item_id], rating)

        # 将数据转换为张量
        self.bx_tensor = torch.tensor(self.bx, dtype=torch.float32)
        self.bi_tensor = torch.tensor(self.bi, dtype=torch.float32)

        # 创建训练集
        self.bx_dataset = TensorDataset(self.bx_tensor)
        self.bi_dataset = TensorDataset(self.bi_tensor)

        # 创建数据加载器
        self.bx_dataloader = DataLoader(self.bx_dataset, batch_size=batch_size, shuffle=True)
        self.bi_dataloader = DataLoader(self.bi_dataset, batch_size=batch_size, shuffle=True)



class CNNBaseline(nn.Module):
    def __init__(self, user_dim, item_dim, embedding_dim, filter_sizes, num_filters):
        super(CNNBaseline, self).__init__()
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

        self.user_convs = nn.ModuleList([
            nn.Conv1d(in_channels=32, out_channels=nf, kernel_size=fs)  # 修改为32以匹配输入数据的通道数
            for fs, nf in zip(self.filter_sizes, self.num_filters)
        ])

        self.item_convs = nn.ModuleList([
            nn.Conv1d(in_channels=32, out_channels=nf, kernel_size=fs)  # 修改为32以匹配输入数据的通道数
            for fs, nf in zip(self.filter_sizes, self.num_filters)
        ])

        self.fc = nn.Linear(self.user_dim + self.item_dim, 1)
    # 其他代码不变
    def forward(self, user_input, item_input):
        # 将用户输入和物品输入拼接在一起
        x = torch.cat([user_input, item_input], dim=0)
        # 通过全连接层
        output = self.fc(x)
        return output
    # def forward(self, user_input, item_input):
    #     # 增加一个通道维度
    #     user_input = user_input.unsqueeze(1)
    #     item_input = item_input.unsqueeze(1)
    #
    #     user_features = [torch.relu(conv(user_input)) for conv in self.user_convs]
    #     user_features = [torch.max_pool1d(feat, feat.size(1)).squeeze(1) for feat in user_features]
    #
    #     item_features = [torch.relu(conv(item_input)) for conv in self.item_convs]
    #     item_features = [torch.max_pool1d(feat, feat.size(1)).squeeze(1) for feat in item_features]
    #
    #     features = torch.cat(user_features + item_features, 0)  # 修改为0以匹配张量的维度
    #     output = self.fc(features.view(features.size(0), -1))  # 使用view方法将特征张量展平
    #     return output

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 创建数据加载器实例
data_loader = DatasetLoader(batch_size=32)

# 创建模型
model = CNNBaseline(user_dim=data_loader.bx_tensor.shape[0],  # 使用shape[0]而不是shape[1]
                    item_dim=data_loader.bi_tensor.shape[0],  # 使用shape[0]而不是shape[1]
                    embedding_dim=32,  # 修改为32以匹配输入数据的通道数
                    filter_sizes=[1],  # 修改为[1]以匹配输入数据的长度
                    num_filters=[100, 100, 100])

def train(model, bx_dataloader, bi_dataloader, epochs):
    # 使用Adam优化器
    optimizer = Adam(model.parameters())
    # 使用均方误差作为损失函数
    criterion = nn.MSELoss()

    # 将模型移动到正确的设备上
    model = model.to(device)

    model.train()  # 将模型设置为训练模式

    for epoch in range(epochs):
        for (bx,), (bi,) in zip(bx_dataloader, bi_dataloader):
            # 将数据移动到正确的设备上
            bx = bx.to(device)
            bi = bi.to(device)

            # 前向传播
            outputs = model(bx, bi)
            # 计算损失
            loss = criterion(outputs, bi)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')

# 训练模型
train(model, data_loader.bx_dataloader, data_loader.bi_dataloader, epochs=10)

# 评���模型
evaluate(model, data_loader.test_dataloader)

# 保存模型
save_model(model, './result/cnn.txt')

# 加载模型
load_model(model, './result/cnn.txt')