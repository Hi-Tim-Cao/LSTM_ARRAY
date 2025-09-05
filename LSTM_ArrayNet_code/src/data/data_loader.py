"""加载MATLAB生成的声呐回波数据"""

import os

import numpy as np

import scipy.io as sio

from torch.utils.data import Dataset



class SonarDataset(Dataset):

    """声呐回波数据集（读取.mat文件）"""

    def __init__(self, data_dir, seq_length=1024):

        self.data_dir = data_dir

        self.seq_length = seq_length  # 固定回波序列长度

        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".mat")]

        if not self.file_list:

            raise ValueError(f"未在{data_dir}找到.mat文件")



    def __len__(self):

        return len(self.file_list)



    def __getitem__(self, idx):

        """获取单个样本：(特征, 标签)"""

        # 加载MATLAB文件

        mat_path = os.path.join(self.data_dir, self.file_list[idx])

        mat_data = sio.loadmat(mat_path)['sim_data']  # 对应MATLAB保存的sim_data结构



        # 提取回波数据（形状：[num_elements, 原始采样点]）

        echo_data = mat_data['echo_data'][0, 0].astype(np.float32)  # 阵元×时间



        # 提取真实坐标（标签）：[num_elements, 2]（X, Y）

        true_coords = mat_data['true_array_pos'][0, 0].astype(np.float32)



        # 提取波形参数（用于构建特征）

        waveform_params = {

            'f0': mat_data['waveform_params'][0, 0]['f0'][0, 0],  # 中心频率

            'bandwidth': mat_data['waveform_params'][0, 0]['B'][0, 0],  # 带宽

            'spacing': mat_data['waveform_params'][0, 0]['d'][0, 0]  # 阵元间距

        }



        # 统一回波序列长度（截断或补零）

        echo_data = self._pad_or_truncate(echo_data)



        # 构建特征向量（回波数据+物理参数）

        features = self._build_features(echo_data, waveform_params)



        return features, true_coords



    def _pad_or_truncate(self, echo_data):

        """将回波序列调整为固定长度"""

        num_elements, current_len = echo_data.shape

        if current_len < self.seq_length:

            # 补零

            pad_length = self.seq_length - current_len

            return np.pad(echo_data, ((0, 0), (0, pad_length)), mode='constant')

        else:

            # 截断

            return echo_data[:, :self.seq_length]



    def _build_features(self, echo_data, params):

        """构建6维特征：时域特征+物理参数"""

        num_elements, seq_len = echo_data.shape

        features = np.zeros((num_elements, seq_len, 6), dtype=np.float32)



        # 特征0-2：时域特征（原始值、幅度、能量）

        features[:, :, 0] = echo_data  # 原始回波

        features[:, :, 1] = np.abs(echo_data)  # 幅度

        features[:, :, 2] = echo_data **2  # 能量



        # 特征3：归一化频率（f0/10000Hz，假设最大频率10kHz）

        features[:, :, 3] = params['f0'] / 10000.0



        # 特征4：归一化带宽（带宽/中心频率）

        features[:, :, 4] = params['bandwidth'] / params['f0'] if params['f0'] != 0 else 0



        # 特征5：归一化阵元间距（间距/波长，波长=1500m/s / f0）

        wavelength = 1500.0 / params['f0'] if params['f0'] != 0 else 1.0

        features[:, :, 5] = params['spacing'] / wavelength



        return features