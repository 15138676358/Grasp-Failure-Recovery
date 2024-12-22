"""
DirNet 模型
Created by Yue Wang on 2024-12-10
Source: Improving the grasping efficiency through experience replay
"""

import torch
import torch.nn as nn


class DirectNet(nn.Module):
    def __init__(self, observation_dim=16, hidden_dim=64, output_dim=1, seq_len=15, num_heads=2, num_layers=3):
        super(DirectNet, self).__init__()
        self.seq_len = seq_len
        self.observation_dim, self.hidden_dim = observation_dim, hidden_dim
        # 视觉预处理
        self.vision_head = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        self.action_head = nn.Sequential(nn.Linear(3, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        # Transformer 编码器层
        self.history_head = nn.Sequential(nn.Linear(4, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc_w = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        self.fc_b = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        # 预测层
        self.prediction = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        # 初始化权重
        self.w, self.b = torch.zeros((2 * hidden_dim)).to(device='cuda'), torch.zeros((2 * hidden_dim)).to(device='cuda')
        
    def forward(self, observations, actions, mode='train'):
        visions = self.vision_head(observations[:, :self.observation_dim])
        actions = self.action_head(actions)
        seq = torch.zeros((observations.size(0), self.seq_len + 1, 4), dtype=torch.float32).to(device=observations.device)
        # 将历史序列数据转换为序列张量
        for i in range(self.seq_len):
            seq[:, i, :] = observations[:, i * 4 + self.observation_dim : i * 4 + self.observation_dim + 4]
        seq = self.history_head(seq)
        # 将视觉特征拼接到序列张量中
        seq[:, -1, :] = visions
        # Transformer 编码
        seq = seq.transpose(0, 1)
        seq = self.transformer_encoder(seq)
        seq = seq.transpose(0, 1)
        # 取最后一个时间步的输出
        seq = seq[:, -1, :]
        # 计算预测层的倍数与偏置
        self.w = 1 * self.fc_w(seq)# + 0.9 * self.w
        self.b = 1 * self.fc_b(seq)# + 0.9 * self.b
        # 计算预测值
        x = self.prediction(torch.cat([visions, actions], dim=1) * self.w + self.b)
        
        return x.squeeze()