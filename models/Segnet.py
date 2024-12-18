"""
SegNet SOTA 模型
Created by Yue Wang on 2024-8-10
Source: 2023_Visuo-Tactile Feedback-Based Robot Manipulation for Object Packing
"""

import torch
import torch.nn as nn


class SegmentNet(nn.Module):
    def __init__(self, hidden_dim=64, mode='train'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.vision_head = nn.Sequential(nn.Linear(16, hidden_dim, bias=False), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU())
        self.history_action_head = nn.Sequential(nn.Linear(3, hidden_dim, bias=False), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU())
        self.history_force_head = nn.Sequential(nn.Linear(1, hidden_dim, bias=False), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU())
        self.merge_layer = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim, bias=False), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim, bias=False), nn.ReLU())
        self.bilinear_layer = nn.Bilinear(in1_features=hidden_dim, in2_features=hidden_dim, out_features=hidden_dim)
        self.position_encoding = self.create_position_encoding().to(device='cuda')
        self.tconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2)
        self.tconv2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2)
        self.tconv3 = nn.ConvTranspose2d(4, 4, kernel_size=2, stride=1)
        self.conv11 = nn.Conv2d(128, 64, kernel_size=2, padding=1)
        self.conv12 = nn.Conv2d(64, 16, kernel_size=2, padding=1)
        self.conv21 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(4, 2, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(2, 1, kernel_size=3, padding=0)
        
        self.conv_layer = nn.Sequential(nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2), 
                                        nn.Conv2d(32, 32, kernel_size=3, padding=1),
                                        nn.Conv2d(32, 16, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
                                        nn.Conv2d(16, 8, kernel_size=3, padding=1),
                                        nn.Conv2d(8, 4, kernel_size=3, padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2),
                                        nn.Conv2d(4, 2, kernel_size=3, padding=1),
                                        nn.Conv2d(2, 1, kernel_size=3, padding=1),
                                        nn.Sigmoid())

        for m in self.modules():
            for linear in m.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.xavier_normal_(linear.weight)

    def create_position_encoding(self):
        pos = torch.arange(0, 121, dtype=torch.float32)
        pos_encoding = pos.reshape(11, 11) / 60 - 1.0

        return pos_encoding


    def forward(self, observations, actions, attempts):
        history_actions = observations[:, 16:].view(-1, 15, 4)[:, :, :3]
        history_forces = observations[:, 16:].view(-1, 15, 4)[:, :, 3:]

        query_feature = self.vision_head(observations[:, :16])
        # query_feature_map = query_feature.view(-1, 1, 11, 11).repeat(1, self.hidden_dim, 1, 1)
        query_feature_map = query_feature.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 11, 11)

        actions_feature = self.history_action_head(history_actions)
        forces_feature = self.history_force_head(history_forces)

        support_feature = torch.cat([actions_feature, forces_feature], dim=-1)
        support_feature = self.merge_layer(support_feature)
        support_feature = torch.mean(support_feature, dim=1) * 15 / attempts.unsqueeze(-1)
        support_feature_map = support_feature.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 11, 11)

        # fusion_feature_map = self.bilinear_layer(query_feature_map.permute(0, 2, 3, 1), support_feature_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        fusion_feature_map = torch.concat([query_feature_map, support_feature_map], dim=1)
        fusion_feature_map *= self.position_encoding

        tx = fusion_feature_map
        tx = self.tconv1(tx)
        tx = self.conv11(tx)
        tx = nn.ReLU()(tx)
        tx = self.conv12(tx)
        tx = nn.ReLU()(tx)
        tx = self.tconv2(tx)
        tx = self.conv21(tx)
        tx = nn.ReLU()(tx)
        tx = self.conv22(tx)
        tx = nn.ReLU()(tx)
        tx = self.tconv3(tx)
        tx = self.conv31(tx)
        tx = nn.ReLU()(tx)
        tx = self.conv32(tx)
        # hotmap = nn.Sigmoid()(tx)
        hotmap = tx

        # 以actions为索引取出hotmap中的值
        x = hotmap[range(hotmap.shape[0]), 0, (actions[:, 0] * 50).long(), (actions[:, 1] * 50).long()]

        if self.mode == 'train':
            return x.squeeze()
        if self.mode == 'deploy':
            return hotmap.squeeze()