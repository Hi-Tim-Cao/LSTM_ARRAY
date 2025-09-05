"""数据加载器构建 - 修复版本"""

import os

import torch

from torch.utils.data import DataLoader, Dataset, random_split

import numpy as np

from src.data.sonar_dataset import SonarDataset

from src.data.preprocessor import DataPreprocessor





class ProcessedDataset(Dataset):

    """修复：经过预处理的数据集包装器"""



    def __init__(self, base_dataset, preprocessor, is_train=True):

        self.base_dataset = base_dataset

        self.preprocessor = preprocessor

        self.is_train = is_train



        # 修复：缓存预处理结果以提高效率

        self.cache_data = []

        self._preprocess_all_data()



    def _preprocess_all_data(self):

        """预处理所有数据并缓存"""

        print(f"预处理 {'训练' if self.is_train else '测试'} 数据...")



        for idx in range(len(self.base_dataset)):

            features, labels = self.base_dataset[idx]



            if self.preprocessor.is_fitted:

                # 已拟合，直接变换

                features_processed, labels_processed = self.preprocessor.transform(features, labels)

            else:

                # 未拟合，返回原始数据（仅在fit阶段）

                features_processed, labels_processed = features, labels.flatten()



            self.cache_data.append((features_processed, labels_processed))



            if idx % 100 == 0:

                print(f"已预处理 {idx}/{len(self.base_dataset)} 个样本")



    def __len__(self):

        return len(self.base_dataset)



    def __getitem__(self, idx):

        """修复：返回缓存的预处理数据"""

        return self.cache_data[idx]





def build_dataloaders(config, is_train=True):

    """修复：构建数据加载器的主函数"""



    # 数据路径

    data_config = config['data']

    if is_train:

        data_path = data_config['train_path']

    else:

        data_path = data_config['test_path']



    print(f"加载数据: {data_path}")



    # 修复：添加数据路径验证

    if not os.path.exists(data_path):

        raise FileNotFoundError(f"数据文件不存在: {data_path}")



    # 创建基础数据集

    base_dataset = SonarDataset(data_path, config)



    # 创建预处理器

    preprocessor_dir = os.path.join(config['paths']['experiments'], 'preprocessor')

    preprocessor = DataPreprocessor(preprocessor_dir)



    if is_train:

        return _build_train_dataloaders(base_dataset, preprocessor, config)

    else:

        return _build_test_dataloader(base_dataset, preprocessor, config)





def _build_train_dataloaders(base_dataset, preprocessor, config):

    """修复：构建训练和验证数据加载器"""



    # 修复：改进验证集划分策略

    total_size = len(base_dataset)

    train_ratio = config['data']['train_split']



    # 避免随机划分导致的时序数据泄露

    use_temporal_split = config['data'].get('temporal_split', False)



    if use_temporal_split:

        # 时序划分：前80%训练，后20%验证

        train_size = int(total_size * train_ratio)

        val_size = total_size - train_size



        train_indices = list(range(train_size))

        val_indices = list(range(train_size, total_size))



        from torch.utils.data import Subset

        train_base_dataset = Subset(base_dataset, train_indices)

        val_base_dataset = Subset(base_dataset, val_indices)

    else:

        # 随机划分（保持原有逻辑）

        train_size = int(total_size * train_ratio)

        val_size = total_size - train_size

        train_base_dataset, val_base_dataset = random_split(

            base_dataset, [train_size, val_size],

            generator=torch.Generator().manual_seed(config['train']['seed'])  # 添加随机种子

        )



    print(f"训练集大小: {len(train_base_dataset)}")

    print(f"验证集大小: {len(val_base_dataset)}")



    # 修复：在训练数据上拟合预处理器

    if not preprocessor.is_fitted:

        try:

            # 检查是否已有预处理器文件

            preprocessor.load_scalers()

            print("加载已有的预处理器")

        except FileNotFoundError:

            print("拟合新的预处理器...")

            preprocessor.fit(train_base_dataset)



    # 创建处理后的数据集

    train_dataset = ProcessedDataset(train_base_dataset, preprocessor, is_train=True)

    val_dataset = ProcessedDataset(val_base_dataset, preprocessor, is_train=True)



    # 修复：改进DataLoader参数

    dataloader_config = config['dataloader']



    # 自动调整num_workers（避免在某些环境下出错）

    import multiprocessing

    max_workers = min(multiprocessing.cpu_count(), 8)  # 限制最大worker数

    num_workers = min(dataloader_config.get('num_workers', 4), max_workers)



    # 检测是否在Windows或特殊环境中（避免multiprocessing问题）

    try:

        if os.name == 'nt':  # Windows

            num_workers = 0

    except:

        num_workers = 0



    train_loader = DataLoader(

        train_dataset,

        batch_size=dataloader_config['batch_size'],

        shuffle=True,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),  # GPU加速

        drop_last=True,  # 确保批次大小一致

        persistent_workers=num_workers > 0  # 持久化workers

    )



    val_loader = DataLoader(

        val_dataset,

        batch_size=dataloader_config['batch_size'],

        shuffle=False,  # 验证集不需要shuffle

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )



    return train_loader, val_loader, preprocessor





def _build_test_dataloader(base_dataset, preprocessor, config):

    """修复：构建测试数据加载器"""



    # 加载预处理器

    try:

        preprocessor.load_scalers()

    except FileNotFoundError:

        raise RuntimeError("未找到预处理器文件，请先进行训练")



    # 创建测试数据集

    test_dataset = ProcessedDataset(base_dataset, preprocessor, is_train=False)



    # 测试数据加载器

    dataloader_config = config['dataloader']



    # 自动调整num_workers

    import multiprocessing

    max_workers = min(multiprocessing.cpu_count(), 8)

    num_workers = min(dataloader_config.get('num_workers', 4), max_workers)



    if os.name == 'nt':  # Windows环境

        num_workers = 0



    test_loader = DataLoader(

        test_dataset,

        batch_size=dataloader_config.get('test_batch_size', dataloader_config['batch_size']),

        shuffle=False,

        num_workers=num_workers,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )



    return test_loader, preprocessor





class BalancedBatchSampler:

    """平衡批次采样器：确保每个批次包含不同类型的样本"""



    def __init__(self, dataset, batch_size, num_replicas=None, rank=None):

        self.dataset = dataset

        self.batch_size = batch_size

        self.num_replicas = num_replicas

        self.rank = rank



        # 可以根据某些特征（如阵元数量、频率等）对样本分组

        self._build_sample_groups()



    def _build_sample_groups(self):

        """根据样本特征构建分组"""

        # 这里可以实现更复杂的样本分组逻辑

        # 例如按频率、阵元数量等特征分组

        pass





class AdaptiveBatchSampler:

    """自适应批次采样器：根据样本难度动态调整采样概率"""



    def __init__(self, dataset, batch_size, loss_history=None):

        self.dataset = dataset

        self.batch_size = batch_size

        self.loss_history = loss_history or {}



        # 初始化采样权重（均匀分布）

        self.sample_weights = np.ones(len(dataset))



    def update_weights(self, batch_indices, batch_losses):

        """根据批次损失更新样本权重"""

        for idx, loss in zip(batch_indices, batch_losses):

            # 损失越大的样本，下次采样概率越高

            self.sample_weights[idx] = 0.9 * self.sample_weights[idx] + 0.1 * loss



    def __iter__(self):

        # 基于权重进行采样

        indices = np.random.choice(

            len(self.dataset),

            size=len(self.dataset),

            p=self.sample_weights / np.sum(self.sample_weights),

            replace=False

        )



        # 按批次大小分组

        for i in range(0, len(indices), self.batch_size):

            batch_indices = indices[i:i+self.batch_size]

            if len(batch_indices) == self.batch_size:

                yield batch_indices



    def __len__(self):

        return len(self.dataset) // self.batch_size





def create_enhanced_dataloaders(config, is_train=True):

    """创建增强版数据加载器，支持更多高级功能"""



    base_dataset = SonarDataset(config['data']['train_path' if is_train else 'test_path'], config)



    preprocessor_dir = os.path.join(config['paths']['experiments'], 'preprocessor')

    preprocessor = DataPreprocessor(preprocessor_dir)



    if is_train:

        # 训练模式的增强功能

        train_size = int(len(base_dataset) * config['data']['train_split'])

        val_size = len(base_dataset) - train_size



        train_base, val_base = random_split(base_dataset, [train_size, val_size])



        # 拟合预处理器

        if not preprocessor.is_fitted:

            preprocessor.fit(train_base)



        # 创建数据集

        train_dataset = ProcessedDataset(train_base, preprocessor, is_train=True)

        val_dataset = ProcessedDataset(val_base, preprocessor, is_train=True)



        # 高级采样策略

        use_adaptive_sampling = config['dataloader'].get('adaptive_sampling', False)



        if use_adaptive_sampling:

            train_sampler = AdaptiveBatchSampler(train_dataset, config['dataloader']['batch_size'])

            train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)

        else:

            train_loader = DataLoader(

                train_dataset,

                batch_size=config['dataloader']['batch_size'],

                shuffle=True,

                num_workers=0,  # 简化处理

                pin_memory=torch.cuda.is_available()

            )



        val_loader = DataLoader(

            val_dataset,

            batch_size=config['dataloader']['batch_size'],

            shuffle=False,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )



        return train_loader, val_loader, preprocessor



    else:

        # 测试模式

        preprocessor.load_scalers()

        test_dataset = ProcessedDataset(base_dataset, preprocessor, is_train=False)



        test_loader = DataLoader(

            test_dataset,

            batch_size=config['dataloader']['batch_size'],

            shuffle=False,

            num_workers=0,

            pin_memory=torch.cuda.is_available()

        )



        return test_loader, preprocessor





class DataLoaderBuilder:

    """数据加载器构建器类：封装所有数据加载逻辑"""



    def __init__(self, config):

        self.config = config

        self.preprocessor = None



    def build_train_loaders(self):

        """构建训练和验证数据加载器"""

        return build_dataloaders(self.config, is_train=True)



    def build_test_loader(self):

        """构建测试数据加载器"""

        return build_dataloaders(self.config, is_train=False)



    def get_data_stats(self):

        """获取数据集统计信息"""

        if self.preprocessor is None:

            raise ValueError("请先构建数据加载器")



        return self.preprocessor.get_feature_stats()



    def validate_data_integrity(self, dataset):

        """验证数据完整性"""

        print("验证数据完整性...")



        for i in range(min(100, len(dataset))):  # 检查前100个样本

            try:

                features, labels = dataset[i]



                # 检查数据形状

                assert features.ndim == 3, f"特征维度错误: {features.shape}"

                assert labels.ndim == 1 or labels.ndim == 2, f"标签维度错误: {labels.shape}"



                # 检查数据类型

                assert not np.isnan(features).any(), f"特征包含NaN: 样本{i}"

                assert not np.isnan(labels).any(), f"标签包含NaN: 样本{i}"



            except Exception as e:

                print(f"数据验证失败，样本{i}: {e}")

                raise



        print("数据完整性验证通过")