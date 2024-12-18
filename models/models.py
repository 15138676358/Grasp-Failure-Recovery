import numpy as np
import torch
import torch.nn as nn


#################### agent 7 ####################
class DirectNet(nn.Module):
    def __init__(self, observation_dim=200, hidden_dim=16, output_dim=1, seq_len=15, num_heads=2, num_layers=3):
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
        self.fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, 2*hidden_dim))
        # 预测层
        self.prediction = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim), nn.Dropout(p=0.1), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
        
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
        # 计算预测层的偏置
        self.b = self.fc(seq)
        # 计算预测值
        x = self.prediction(torch.cat([visions, actions], dim=1) + self.b)
        
        return x.squeeze()

    
#################### agent 6 ####################
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
#################### aborted ####################
class RepNet(nn.Module):
    def __init__(self, hidden_dim=16, output_dim=1, seq_len=15, num_heads=2, num_layers=2):
        super(RepNet, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        # 视觉预处理
        self.vision_head = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(p=0.001), nn.Linear(32, hidden_dim))
        # Transformer 编码器层
        self.embedding = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Dropout(p=0.001), nn.Linear(32, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(p=0.001), nn.Linear(32, output_dim))
        for m in self.modules():
            for linear in m.modules():
                if isinstance(linear, nn.Linear):
                    nn.init.xavier_normal_(linear.weight)
        
    def forward(self, inputs, mode='train'):
        observations = inputs
        seq = torch.zeros((observations.size(0), self.seq_len + 1, 4), dtype=torch.float32).to(device=observations.device)
        # 将历史序列数据转换为序列张量
        for i in range(self.seq_len):
            seq[:, i, :] = observations[:, i * 4 + 16:i * 4 + 20]
        seq = self.embedding(seq)
        # 将视觉特征和动作张量拼接到序列张量中
        seq[:, -1, :] = self.vision_head(observations[:, :16])
        # seq[:, :-2, :] = 0
        # Transformer 编码
        seq = self.transformer_encoder(seq)
        # 取最后一个时间步的输出
        seq = seq[:, -1, :]
        x = self.fc(seq)
        return seq, x
    
class DyNet(nn.Module):
    def __init__(self, input_dim=79, hidden_dim=128, output_dim=1):
        super(DyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=0.05)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        # x = torch.relu(self.fc3(x))
        # x = self.dropout(x)
        x = self.fc4(x)

        return x.squeeze()

class InfoMaxNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=16):
        super(InfoMaxNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(output_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        z = self.encoder(x)
        return z

    def discriminate(self, z, x):
        combined = torch.cat([z, x], dim=1)
        score = self.discriminator(combined)
        return score
    
    def cal_mutual_information_loss(self, z, x):
        batch_size = z.size(0)
        positive_score = self.discriminate(z, x)
        
        # 生成负样本
        permuted_indices = torch.randperm(batch_size)
        negative_score = self.discriminate(z, x[permuted_indices])
        
        # 计算正样本和负样本的损失
        positive_loss = -torch.mean(torch.log(torch.sigmoid(positive_score)))
        negative_loss = -torch.mean(torch.log(1 - torch.sigmoid(negative_score)))
        
        return positive_loss + negative_loss
    
# 定义卷积神经网络模型
class CNNModel(nn.Module):
    # 定义模型结构, input为16维的视觉和15，3的历史序列数据，首先讲视觉和序列展平拼接为61维向量，然后经过MLP输出一个32通道的1*1数据，再通过反卷积输出一个1通道的50*50热力图
    def __init__(self):
        super(CNNModel, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(61, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        self.deconv = nn.Sequential(
            # 输入尺寸：(16, 4, 4)
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # 输出尺寸: (8, 8, 8)
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 输出尺寸: (4, 16, 16)
            nn.ConvTranspose2d(4, 1, kernel_size=5, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            # 输出尺寸: (1, 25, 25)
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            # 输出尺寸: (1, 50, 50)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        observations, actions = inputs[:, :61], inputs[:, 61:]
        embeddings = self.embedding(observations)
        embeddings = embeddings.view(-1, 16, 4, 4)
        hotmap = 4 * self.deconv(embeddings)
        # 根据actions从x中取force值
        forces = hotmap[range(hotmap.shape[0]), 0, (actions[:, 0] * 50).long(), (actions[:, 1] * 50).long()]

        return forces
#################### agent 5 ####################
# 定义 Transformer 模型
class TransformerNet(nn.Module):
    def __init__(self, observation_dim=200, hidden_dim=16, output_dim=1, seq_len=15, num_heads=2, num_layers=3):
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

# 定义 SegmentNet SOTA 模型    
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


    def forward(self, inputs, attempts):
        observations, actions = inputs[:, :76], inputs[:, 76:]
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

   
if __name__ == '__main__':
    pass
