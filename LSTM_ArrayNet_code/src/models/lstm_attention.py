"""LSTM+注意力模型（核心模型）- 修复版本"""

import torch

import torch.nn as nn

import torch.nn.functional as F





class LSTMAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.num_elements = config['data']['num_elements']  # 阵元数量



        # LSTM参数

        input_dim = config['model']['input_dim']

        hidden_dim = config['model']['hidden_dim']

        num_layers = config['model']['num_layers']

        dropout = config['model']['dropout']

        bidirectional = config['model']['bidirectional']



        # 定义LSTM层

        self.lstm = nn.LSTM(

            input_size=input_dim,

            hidden_size=hidden_dim,

            num_layers=num_layers,

            batch_first=True,

            dropout=dropout if num_layers > 1 else 0,

            bidirectional=bidirectional

        )



        # LSTM输出维度（双向则乘2）

        self.lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim



        # 修复：重新设计注意力机制 - 为每个阵元独立建模

        attention_dim = config['model']['attention_dim']



        # 阵元级注意力（计算每个阵元内部时间步的重要性）

        self.temporal_attention = nn.Sequential(

            nn.Linear(self.lstm_out_dim, attention_dim),

            nn.Tanh(),

            nn.Linear(attention_dim, 1)

        )



        # 阵元间注意力（计算不同阵元的重要性）

        self.element_attention = nn.Sequential(

            nn.Linear(self.lstm_out_dim, attention_dim),

            nn.Tanh(),

            nn.Linear(attention_dim, 1)

        )



        # 修复：为每个阵元分别预测坐标

        self.element_predictors = nn.ModuleList([

            nn.Sequential(

                nn.Linear(self.lstm_out_dim, 256),

                nn.ReLU(),

                nn.Dropout(dropout),

                nn.Linear(256, 128),

                nn.ReLU(),

                nn.Linear(128, 2)  # 预测该阵元的X,Y坐标

            ) for _ in range(self.num_elements)

        ])



        # 可选：全局上下文建模（考虑阵元间相互关系）

        self.use_global_context = config['model'].get('use_global_context', True)

        if self.use_global_context:

            self.global_context = nn.Sequential(

                nn.Linear(self.lstm_out_dim * self.num_elements, 512),

                nn.ReLU(),

                nn.Dropout(dropout),

                nn.Linear(512, self.lstm_out_dim)

            )



    def forward(self, x):

        """

        :param x: 输入特征，形状 [batch_size, num_elements, seq_len, input_dim]

        :return: 预测坐标，形状 [batch_size, num_elements×2]

        """

        batch_size, num_elems, seq_len, input_dim = x.shape



        # 合并batch和阵元维度，适应LSTM输入要求

        x_reshaped = x.reshape(batch_size * num_elems, seq_len, input_dim)  # [B×N, T, F]



        # LSTM特征提取

        lstm_out, _ = self.lstm(x_reshaped)  # [B×N, T, lstm_out_dim]



        # 恢复阵元维度

        lstm_out = lstm_out.reshape(batch_size, num_elems, seq_len, self.lstm_out_dim)  # [B, N, T, D]



        # 修复：改进的注意力机制

        element_features = []



        for i in range(num_elems):

            # 对每个阵元计算时间注意力

            element_lstm = lstm_out[:, i, :, :]  # [B, T, D]



            # 时间维度注意力权重

            temporal_attn = self.temporal_attention(element_lstm)  # [B, T, 1]

            temporal_weights = F.softmax(temporal_attn, dim=1)  # [B, T, 1]



            # 时间维度加权平均

            element_feature = torch.sum(temporal_weights * element_lstm, dim=1)  # [B, D]

            element_features.append(element_feature)



        # 堆叠所有阵元特征

        element_features = torch.stack(element_features, dim=1)  # [B, N, D]



        # 可选：考虑全局上下文

        if self.use_global_context:

            # 计算阵元间注意力

            element_attn = self.element_attention(element_features)  # [B, N, 1]

            element_weights = F.softmax(element_attn, dim=1)  # [B, N, 1]



            # 全局上下文特征

            global_context = torch.sum(element_weights * element_features, dim=1)  # [B, D]

            global_context = self.global_context(

                element_features.reshape(batch_size, -1)

            )  # [B, D]

            global_context = global_context.unsqueeze(1).expand(-1, num_elems, -1)  # [B, N, D]



            # 融合局部和全局特征

            element_features = element_features + 0.1 * global_context  # 残差连接



        # 修复：为每个阵元独立预测坐标

        coordinates = []

        for i in range(num_elems):

            coord = self.element_predictors[i](element_features[:, i, :])  # [B, 2]

            coordinates.append(coord)



        # 拼接所有坐标 [B, N, 2] -> [B, N×2]

        coordinates = torch.stack(coordinates, dim=1)  # [B, N, 2]

        coordinates = coordinates.reshape(batch_size, -1)  # [B, N×2]



        return coordinates



    def get_attention_weights(self, x):

        """获取注意力权重用于可视化分析"""

        batch_size, num_elems, seq_len, input_dim = x.shape

        x_reshaped = x.reshape(batch_size * num_elems, seq_len, input_dim)

        lstm_out, _ = self.lstm(x_reshaped)

        lstm_out = lstm_out.reshape(batch_size, num_elems, seq_len, self.lstm_out_dim)



        temporal_weights = []

        element_weights = []



        # 计算时间注意力权重

        for i in range(num_elems):

            element_lstm = lstm_out[:, i, :, :]

            temporal_attn = self.temporal_attention(element_lstm)

            temporal_weights.append(F.softmax(temporal_attn, dim=1))



        temporal_weights = torch.stack(temporal_weights, dim=1)  # [B, N, T, 1]



        # 计算阵元注意力权重（如果使用全局上下文）

        if self.use_global_context:

            element_features = []

            for i in range(num_elems):

                element_lstm = lstm_out[:, i, :, :]

                temporal_attn = temporal_weights[:, i, :, :]

                element_feature = torch.sum(temporal_attn * element_lstm, dim=1)

                element_features.append(element_feature)



            element_features = torch.stack(element_features, dim=1)

            element_attn = self.element_attention(element_features)

            element_weights = F.softmax(element_attn, dim=1)



        return {

            'temporal_weights': temporal_weights,

            'element_weights': element_weights if element_weights else None

        }





class SimpleLSTMAttention(nn.Module):

    """简化版本的LSTM注意力模型（用于对比）"""

    def __init__(self, config):

        super().__init__()

        self.num_elements = config['data']['num_elements']



        input_dim = config['model']['input_dim']

        hidden_dim = config['model']['hidden_dim']



        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

        lstm_out_dim = hidden_dim * 2



        # 简单的全连接预测层

        self.predictor = nn.Sequential(

            nn.Linear(lstm_out_dim, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, self.num_elements * 2)

        )



    def forward(self, x):

        batch_size, num_elems, seq_len, input_dim = x.shape



        # 只使用第一个阵元的数据（简化处理）

        x_first = x[:, 0, :, :]  # [B, T, F]



        lstm_out, _ = self.lstm(x_first)  # [B, T, D]

        last_hidden = lstm_out[:, -1, :]  # [B, D]



        coords = self.predictor(last_hidden)  # [B, N×2]

        return coords