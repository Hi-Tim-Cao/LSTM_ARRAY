"""数据预处理工具（归一化、特征转换）- 修复版本"""

import os

import numpy as np

import pickle

from sklearn.preprocessing import MinMaxScaler, StandardScaler





class DataPreprocessor:

    """用于特征和标签的归一化处理 - 修复版本"""



    def __init__(self, save_dir, feature_range=(-1, 1), label_range=(0, 1),

                 normalize_method='minmax'):

        self.save_dir = save_dir

        os.makedirs(save_dir, exist_ok=True)



        self.feature_range = feature_range

        self.label_range = label_range

        self.normalize_method = normalize_method



        # 修复：改进归一化器初始化

        if normalize_method == 'minmax':

            self.scaler_features = MinMaxScaler(feature_range=feature_range)

            self.scaler_labels = MinMaxScaler(feature_range=label_range)

        elif normalize_method == 'standard':

            self.scaler_features = StandardScaler()

            self.scaler_labels = StandardScaler()

        else:

            raise ValueError(f"不支持的归一化方法: {normalize_method}")



        self.is_fitted = False



    def fit(self, dataset):

        """修复：改进的拟合过程"""

        print("开始拟合预处理器...")



        # 收集所有特征和标签

        all_features = []

        all_labels = []



        for idx, (features, labels) in enumerate(dataset):

            # features: [num_elements, seq_len, feature_dim]

            # labels: [num_elements, 2]



            # 修复：正确处理特征维度

            # 将特征展平用于拟合归一化器

            features_flat = features.reshape(-1, features.shape[-1])  # [N×T, F]

            all_features.append(features_flat)



            # 修复：正确处理标签维度

            labels_flat = labels.reshape(-1)  # [N×2] -> [N*2]

            all_labels.append(labels_flat)



            if idx % 100 == 0:

                print(f"已处理 {idx} 个样本")



        # 合并所有数据

        all_features = np.vstack(all_features)  # [所有时间步×特征数]

        all_labels = np.concatenate(all_labels)  # [所有坐标值]



        print(f"特征形状: {all_features.shape}")

        print(f"标签形状: {all_labels.shape}")



        # 修复：改进标签归一化策略

        # 为了保持坐标的相对关系，分别对X和Y坐标归一化

        all_labels_2d = all_labels.reshape(-1, 2)  # [N*num_elements, 2]



        # 拟合归一化器

        self.scaler_features.fit(all_features)



        # 分别对X和Y坐标拟合归一化器

        self.scaler_labels_x = MinMaxScaler(feature_range=self.label_range)

        self.scaler_labels_y = MinMaxScaler(feature_range=self.label_range)



        self.scaler_labels_x.fit(all_labels_2d[:, 0:1])  # X坐标

        self.scaler_labels_y.fit(all_labels_2d[:, 1:2])  # Y坐标



        self.is_fitted = True



        # 保存预处理器状态

        self._save_scalers()

        print("预处理器拟合完成并已保存")



    def transform(self, features, labels):

        """修复：改进的变换方法"""

        if not self.is_fitted:

            raise ValueError("预处理器尚未拟合，请先调用fit方法")



        # 特征变换

        original_shape = features.shape  # [num_elements, seq_len, feature_dim]

        features_flat = features.reshape(-1, features.shape[-1])  # [N×T, F]

        features_scaled = self.scaler_features.transform(features_flat)

        features_scaled = features_scaled.reshape(original_shape)  # 恢复原始形状



        # 标签变换 - 修复：分别处理X和Y坐标

        num_elements = labels.shape[0]

        labels_flat = labels.reshape(-1, 2)  # [num_elements, 2]



        # 分别归一化X和Y坐标

        x_scaled = self.scaler_labels_x.transform(labels_flat[:, 0:1])

        y_scaled = self.scaler_labels_y.transform(labels_flat[:, 1:2])



        labels_scaled = np.concatenate([x_scaled, y_scaled], axis=1)  # [num_elements, 2]

        labels_scaled = labels_scaled.flatten()  # [num_elements×2]



        return features_scaled, labels_scaled



    def inverse_transform_labels(self, labels_scaled):

        """修复：改进的标签反归一化"""

        if not self.is_fitted:

            raise ValueError("预处理器尚未拟合")



        # labels_scaled可能是 [batch_size, num_elements×2] 或 [num_elements×2]

        original_shape = labels_scaled.shape



        if labels_scaled.ndim == 1:

            # 单样本情况

            labels_2d = labels_scaled.reshape(-1, 2)  # [num_elements, 2]

        else:

            # 批次情况

            labels_2d = labels_scaled.reshape(-1, 2)  # [batch_size×num_elements, 2]



        # 分别反归一化X和Y坐标

        x_orig = self.scaler_labels_x.inverse_transform(labels_2d[:, 0:1])

        y_orig = self.scaler_labels_y.inverse_transform(labels_2d[:, 1:2])



        labels_orig = np.concatenate([x_orig, y_orig], axis=1)



        # 恢复原始形状

        if len(original_shape) == 1:

            return labels_orig.flatten()

        else:

            return labels_orig.reshape(original_shape)



    def inverse_transform_features(self, features_scaled):

        """特征反归一化"""

        if not self.is_fitted:

            raise ValueError("预处理器尚未拟合")



        original_shape = features_scaled.shape

        features_flat = features_scaled.reshape(-1, features_scaled.shape[-1])

        features_orig = self.scaler_features.inverse_transform(features_flat)

        return features_orig.reshape(original_shape)



    def _save_scalers(self):

        """保存归一化器状态"""

        scalers_path = os.path.join(self.save_dir, 'scalers.pkl')

        scalers_data = {

            'scaler_features': self.scaler_features,

            'scaler_labels_x': self.scaler_labels_x,

            'scaler_labels_y': self.scaler_labels_y,

            'feature_range': self.feature_range,

            'label_range': self.label_range,

            'normalize_method': self.normalize_method,

            'is_fitted': self.is_fitted

        }



        with open(scalers_path, 'wb') as f:

            pickle.dump(scalers_data, f)



    def load_scalers(self):

        """加载归一化器状态"""

        scalers_path = os.path.join(self.save_dir, 'scalers.pkl')



        if not os.path.exists(scalers_path):

            raise FileNotFoundError(f"未找到预处理器文件: {scalers_path}")



        with open(scalers_path, 'rb') as f:

            scalers_data = pickle.load(f)



        self.scaler_features = scalers_data['scaler_features']

        self.scaler_labels_x = scalers_data['scaler_labels_x']

        self.scaler_labels_y = scalers_data['scaler_labels_y']

        self.feature_range = scalers_data['feature_range']

        self.label_range = scalers_data['label_range']

        self.normalize_method = scalers_data['normalize_method']

        self.is_fitted = scalers_data['is_fitted']



        print(f"已加载预处理器状态: {scalers_path}")



    def get_feature_stats(self):

        """获取特征统计信息"""

        if not self.is_fitted:

            raise ValueError("预处理器尚未拟合")



        stats = {

            'feature_min': self.scaler_features.data_min_,

            'feature_max': self.scaler_features.data_max_,

            'feature_range': self.scaler_features.data_range_,

        }



        if hasattr(self, 'scaler_labels_x'):

            stats.update({

                'label_x_min': self.scaler_labels_x.data_min_[0],

                'label_x_max': self.scaler_labels_x.data_max_[0],

                'label_y_min': self.scaler_labels_y.data_min_[0],

                'label_y_max': self.scaler_labels_y.data_max_[0],

            })



        return stats





class EnhancedDataPreprocessor(DataPreprocessor):

    """增强版数据预处理器，支持更多功能"""



    def __init__(self, save_dir, **kwargs):

        super().__init__(save_dir, **kwargs)



        # 额外的预处理选项

        self.remove_outliers = kwargs.get('remove_outliers', False)

        self.outlier_threshold = kwargs.get('outlier_threshold', 3.0)  # 3倍标准差

        self.smooth_signals = kwargs.get('smooth_signals', False)



    def fit(self, dataset):

        """增强的拟合过程，支持异常值检测"""

        print("开始拟合增强预处理器...")



        all_features = []

        all_labels = []

        outlier_indices = []



        for idx, (features, labels) in enumerate(dataset):

            features_flat = features.reshape(-1, features.shape[-1])

            labels_flat = labels.reshape(-1)



            # 异常值检测

            if self.remove_outliers:

                # 基于Z-score检测特征异常值

                z_scores = np.abs((features_flat - np.mean(features_flat, axis=0)) /

                                 (np.std(features_flat, axis=0) + 1e-8))

                if np.any(z_scores > self.outlier_threshold):

                    outlier_indices.append(idx)

                    continue



            all_features.append(features_flat)

            all_labels.append(labels_flat)



            if idx % 100 == 0:

                print(f"已处理 {idx} 个样本")



        if outlier_indices:

            print(f"检测到 {len(outlier_indices)} 个异常样本并已移除")



        # 调用父类的拟合逻辑

        super().fit([(np.vstack(all_features), np.concatenate(all_labels))])



    def transform(self, features, labels):

        """增强的变换方法，支持信号平滑"""

        features_scaled, labels_scaled = super().transform(features, labels)



        # 可选的信号平滑

        if self.smooth_signals:

            features_scaled = self._smooth_signal(features_scaled)



        return features_scaled, labels_scaled



    def _smooth_signal(self, signal, window_size=3):

        """简单的滑动平均平滑"""

        from scipy import ndimage

        return ndimage.uniform_filter1d(signal, size=window_size, axis=1)