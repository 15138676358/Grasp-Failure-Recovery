"""
TransNet 模型
Created by Yue Wang on 2024-9-10
Source: Learning from Failures for Improving the Generality of Grasping Failure Recovery
"""

import torch
import torch.nn as nn


class TransformerNet(nn.Module):
    def __init__(self, observation_dim=16, hidden_dim=64, output_dim=1, seq_len=15, num_heads=2, num_layers=3):
        super(TransformerNet, self).__init__()
        self.seq_len = seq_len
        self.observation_dim, self.hidden_dim = observation_dim, hidden_dim
        # 视觉预处理
        self.vision_head = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        self.action_head = nn.Sequential(nn.Linear(3, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        # Transformer 编码器层
        self.embedding = nn.Sequential(nn.Linear(4, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        # for m in self.modules():
        #     for linear in m.modules():
        #         if isinstance(linear, nn.Linear):
        #             nn.init.xavier_normal_(linear.weight)
        
    def forward(self, observations, actions, mode='train'):
        seq = torch.zeros((observations.size(0), self.seq_len + 2, 4), dtype=torch.float32).to(device=observations.device)
        # 将历史序列数据转换为序列张量
        for i in range(self.seq_len):
            seq[:, i, :] = observations[:, i * 4 + self.observation_dim : i * 4 + self.observation_dim + 4]
        seq = self.embedding(seq)
        # 将视觉特征和动作张量拼接到序列张量中
        seq[:, -2, :] = self.vision_head(observations[:, :self.observation_dim])
        seq[:, -1, :] = self.action_head(actions)
        # seq[:, :-2, :] = 0
        # Transformer 编码
        seq = seq.transpose(0, 1)
        seq = self.transformer_encoder(seq)
        seq = seq.transpose(0, 1)
        # 取最后一个时间步的输出
        seq = seq[:, -1, :]
        if mode == 'train':
            x = self.fc(seq)
        if mode == 'deploy':
            x = seq
        
        return x.squeeze()