"""模型训练器 - 修复版本"""

import os

import time

import numpy as np  # 修复：添加缺失的numpy导入

import torch

import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR

from datetime import datetime

from src.utils.logger import Logger

from src.utils.metrics import calc_rmse

from src.models.lstm_attention import LSTMAttention

from .loss import SonarLoss





class Trainer:

    def __init__(self, config):

        self.config = config

        self.device = torch.device(config['train']['device'])

        self.epochs = config['train']['epochs']

        self.save_interval = config['train']['save_interval']



        # 创建实验目录（按时间命名）

        self.exp_name = f"{datetime.now().strftime('%Y%m%d_%H%M')}_{config['model']['type']}"

        self.exp_dir = os.path.join(config['paths']['experiments'], self.exp_name)

        os.makedirs(self.exp_dir, exist_ok=True)

        self.weights_dir = os.path.join(self.exp_dir, 'weights')

        os.makedirs(self.weights_dir, exist_ok=True)



        # 初始化日志

        self.logger = Logger(os.path.join(self.exp_dir, 'logs'))



        # 初始化模型、损失函数、优化器

        self.model = LSTMAttention(config).to(self.device)

        self.criterion = SonarLoss(config)

        self.optimizer = self._get_optimizer()

        self.scheduler = self._get_scheduler()



        # 记录最佳验证指标

        self.best_val_rmse = float('inf')



    def _get_optimizer(self):

        """获取优化器"""

        opt_cfg = self.config['optimizer']

        if opt_cfg['type'] == 'AdamW':

            return optim.AdamW(

                self.model.parameters(),

                lr=opt_cfg['lr'],

                weight_decay=opt_cfg['weight_decay']

            )

        return optim.Adam(self.model.parameters(), lr=opt_cfg['lr'])



    def _get_scheduler(self):

        """获取学习率调度器 - 修复T_max设置"""

        sched_cfg = self.config['scheduler']

        if sched_cfg['type'] == 'CosineAnnealingLR':

            # 修复：T_max应该等于总训练epochs，避免学习率过早降到最小值

            T_max = min(sched_cfg.get('T_max', self.epochs), self.epochs)

            return CosineAnnealingLR(

                self.optimizer,

                T_max=T_max,

                eta_min=1e-6

            )

        return None



    def train_epoch(self, train_loader):

        """训练一个epoch"""

        self.model.train()

        total_loss = 0.0

        loss_components = {'loss_pos': 0.0, 'loss_spacing': 0.0}



        for batch_idx, (features, labels) in enumerate(train_loader):

            # 数据移至设备

            features = features.to(self.device, non_blocking=True)  # 添加non_blocking优化

            labels = labels.to(self.device, non_blocking=True)



            # 前向传播

            self.optimizer.zero_grad()

            outputs = self.model(features)

            loss, components = self.criterion(outputs, labels)



            # 修复：添加梯度裁剪防止梯度爆炸

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()



            # 累计损失

            total_loss += loss.item()

            for k, v in components.items():

                loss_components[k] += v



            # 打印批次日志

            if batch_idx % 10 == 0:

                self.logger.log_batch(

                    batch_idx, len(train_loader), loss.item(), components

                )



        # 计算平均损失

        avg_loss = total_loss / len(train_loader)

        for k in loss_components:

            loss_components[k] /= len(train_loader)

        return avg_loss, loss_components



    def validate(self, val_loader, preprocessor):

        """验证模型性能"""

        self.model.eval()

        total_loss = 0.0

        all_preds = []

        all_labels = []



        with torch.no_grad():

            for features, labels in val_loader:

                features = features.to(self.device, non_blocking=True)

                labels = labels.to(self.device, non_blocking=True)



                outputs = self.model(features)

                loss, _ = self.criterion(outputs, labels)

                total_loss += loss.item()



                # 保存预测和标签（用于计算指标）

                all_preds.append(outputs.cpu().numpy())

                all_labels.append(labels.cpu().numpy())



        # 计算平均损失和RMSE

        avg_loss = total_loss / len(val_loader)



        # 修复：正确处理预测结果的维度

        all_preds = np.vstack(all_preds)  # [N_samples, num_elements*2]

        all_labels = np.vstack(all_labels)  # [N_samples, num_elements*2]



        # 反归一化

        all_preds_orig = preprocessor.inverse_transform_labels(all_preds)

        all_labels_orig = preprocessor.inverse_transform_labels(all_labels)



        rmse = calc_rmse(all_preds_orig, all_labels_orig)



        return avg_loss, rmse



    def train(self, train_loader, val_loader, preprocessor):

        """完整训练流程"""

        start_time = time.time()

        print(f"开始训练，实验目录：{self.exp_dir}")



        # 修复：添加异常处理

        try:

            for epoch in range(1, self.epochs + 1):

                print(f"\n===== Epoch {epoch}/{self.epochs} =====")



                # 训练

                train_loss, train_components = self.train_epoch(train_loader)

                self.logger.log_train(epoch, train_loss, train_components)



                # 验证

                val_loss, val_rmse = self.validate(val_loader, preprocessor)

                self.logger.log_val(epoch, val_loss, val_rmse)



                # 学习率调度

                if self.scheduler:

                    self.scheduler.step()



                # 保存模型

                if epoch % self.save_interval == 0:

                    torch.save(

                        self.model.state_dict(),

                        os.path.join(self.weights_dir, f"epoch_{epoch}.pth")

                    )



                # 保存最佳模型

                if val_rmse < self.best_val_rmse:

                    self.best_val_rmse = val_rmse

                    torch.save(

                        self.model.state_dict(),

                        os.path.join(self.weights_dir, "best_model.pth")

                    )

                    print(f"保存最佳模型（RMSE: {val_rmse:.4f}m）")



        except KeyboardInterrupt:

            print("\n训练被用户中断")

        except Exception as e:

            print(f"\n训练过程中出现错误: {e}")

            raise

        finally:

            # 训练结束清理

            total_time = time.time() - start_time

            print(f"\n训练完成！总耗时：{total_time:.2f}秒")

            print(f"最佳验证RMSE：{self.best_val_rmse:.4f}m")

            self.logger.close()





def start_training(config_path):

    """训练入口函数"""

    from src.utils.config import load_config

    from src.data.dataloader_builder import build_dataloaders



    # 加载配置

    config = load_config(config_path)



    # 构建数据加载器

    train_loader, val_loader, preprocessor = build_dataloaders(config, is_train=True)



    # 启动训练

    trainer = Trainer(config)

    trainer.train(train_loader, val_loader, preprocessor)