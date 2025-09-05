"""训练日志工具（控制台+TensorBoard）"""

import os

import time

from tensorboardX import SummaryWriter



class Logger:

    def __init__(self, log_dir):

        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir)

        self.start_time = time.time()



    def log_batch(self, batch_idx, total_batches, loss, components):

        """记录批次日志"""

        elapsed = time.time() - self.start_time

        print(f"Batch {batch_idx+1}/{total_batches} | Loss: {loss:.6f} | "

              f"位置损失: {components['loss_pos']:.6f} | "

              f"间距损失: {components['loss_spacing']:.6f} | "

              f"耗时: {elapsed:.2f}s")



    def log_train(self, epoch, loss, components):

        """记录训练epoch日志"""

        print(f"【训练】Epoch {epoch} | 平均损失: {loss:.6f} | "

              f"位置损失: {components['loss_pos']:.6f} | "

              f"间距损失: {components['loss_spacing']:.6f}")

        # 写入TensorBoard

        self.writer.add_scalar('train/loss_total', loss, epoch)

        for k, v in components.items():

            self.writer.add_scalar(f'train/{k}', v, epoch)



    def log_val(self, epoch, loss, rmse):

        """记录验证epoch日志"""

        print(f"【验证】Epoch {epoch} | 平均损失: {loss:.6f} | "

              f"RMSE: {rmse:.4f}m")

        # 写入TensorBoard

        self.writer.add_scalar('val/loss', loss, epoch)

        self.writer.add_scalar('val/rmse', rmse, epoch)



    def close(self):

        """关闭日志写入器"""

        self.writer.close()