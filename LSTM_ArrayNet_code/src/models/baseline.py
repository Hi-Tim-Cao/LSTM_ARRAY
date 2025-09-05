"""基线模型（用于对比）"""

import torch

import torch.nn as nn





class MLPBaseline(nn.Module):

    """简单MLP基线模型"""



    def __init__(self, config):

        super().__init__()

        num_elements = config['data']['num_elements']

        seq_len = config['data']['seq_length']

        input_dim = config['model']['input_dim']



        # 输入维度：阵元数×序列长度×特征维度

        input_size = num_elements * seq_len * input_dim



        self.fc = nn.Sequential(

            nn.Linear(input_size, 1024),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Linear(512, num_elements * 2)

        )



    def forward(self, x):

        # 展平输入：[B, N, T, F] → [B, N*T*F]

        x_flat = x.flatten(start_dim=1)

        return self.fc(x_flat)