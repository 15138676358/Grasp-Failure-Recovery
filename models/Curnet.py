"""
CurNet 模型
Created by Yue Wang on 2024-11-10
Source: A Self-supervised Framework For Data Efficient Grasping Failure Recovery Learning
"""

import torch
import torch.nn as nn

class FeatureNet(nn.Module):
    def __init__(self, observation_dim=200, hidden_dim=32, seq_len=15, num_heads=2, num_layers=2):
        super(FeatureNet, self).__init__()
        self.seq_len = seq_len
        self.observation_dim, self.hidden_dim = observation_dim, hidden_dim
        # 视觉预处理
        self.embedding_vision = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        # Transformer 编码器层
        self.embedding_seq = nn.Sequential(nn.Linear(4, hidden_dim), nn.Dropout(0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))
        self.embedding_action = nn.Sequential(nn.Linear(3, hidden_dim), nn.Dropout(0.1), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.1), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, observations, actions, mode='encode'):
        seq = torch.zeros((observations.size(0), self.seq_len + 2, 4), dtype=torch.float32).to(device=observations.device)
        # 将历史序列数据转换为序列张量
        for i in range(self.seq_len):
            seq[:, i, :] = observations[:, i * 4 + self.observation_dim : i * 4 + self.observation_dim + 4]
        seq = self.embedding_seq(seq)
        # 将视觉特征和动作张量拼接到序列张量中
        seq[:, -1, :] = self.embedding_vision(observations[:, :self.observation_dim])
        if mode == 'direct':
            seq[:, -2, :] = self.embedding_action(actions)
        # Transformer 编码
        seq = seq.transpose(0, 1)
        seq = self.transformer_encoder(seq)
        seq = seq.transpose(0, 1)
        # 取最后一个时间步的输出
        seq = seq[:, -1, :]
        if mode == 'encode':
            return seq
        if mode == 'direct':
            x = self.fc(seq)
            return x

class CuriosityNet(nn.Module):
    def __init__(self, observation_dim=200, action_dim=3, hidden_dim=32, measure_dim=1):
        super(CuriosityNet, self).__init__()
        # self.feature_module = nn.Sequential(
        #     nn.Linear(state_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        self.feature_module = FeatureNet(observation_dim=observation_dim, hidden_dim=hidden_dim)
        self.direct_module = FeatureNet(observation_dim=observation_dim, hidden_dim=hidden_dim, num_heads=2, num_layers=3)
        self.decode_module = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.001),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, measure_dim)
        )
        self.forward_module = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.001),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.inverse_module = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.001),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observations, actions, next_observations, mode='train'):
        state = self.feature_module(observations, actions)
        next_state = self.feature_module(next_observations, actions)

        pred_next_state = self.forward_module(torch.cat([state, actions], dim=1))
        pred_measure = self.decode_module(pred_next_state)
        pred_action = self.inverse_module(torch.cat([state, next_state], dim=1))
        direct_measure = self.direct_module(observations, actions, mode='direct')
        
        if mode == 'train':
            return next_state, pred_next_state, pred_measure, pred_action, direct_measure
        if mode == 'deploy':
            return pred_next_state

class ValueNet(nn.Module):
    def __init__(self, input_dim=79, hidden_dim=64, output_dim=1):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        # # Xavier初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        # # batch normalization
        # self.bn = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x.squeeze()