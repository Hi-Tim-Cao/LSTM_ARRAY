"""损失函数定义 - 修复版本"""

import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np





class SonarLoss(nn.Module):

    """修复：改进的声呐阵列位置估计损失函数"""



    def __init__(self, config):

        super().__init__()

        self.num_elements = config['data']['num_elements']



        # 损失权重

        loss_config = config['loss']

        self.w_pos = loss_config['weight_position']

        self.w_spacing = loss_config.get('weight_spacing', 0.1)

        self.w_shape = loss_config.get('weight_shape', 0.05)  # 新增：形状保持损失

        self.w_smooth = loss_config.get('weight_smooth', 0.01)  # 新增：平滑性损失



        # 其他参数

        self.use_adaptive_weights = loss_config.get('adaptive_weights', False)

        self.loss_type = loss_config.get('type', 'mse')  # 支持不同损失类型



    def forward(self, pred, target):

        """

        :param pred: 预测坐标 [batch_size, num_elements×2]

        :param target: 真实坐标 [batch_size, num_elements×2]

        :return: 总损失, 损失组件字典

        """

        batch_size = pred.size(0)



        # 重塑为坐标形式 [batch_size, num_elements, 2]

        pred_coords = pred.reshape(batch_size, self.num_elements, 2)

        target_coords = target.reshape(batch_size, self.num_elements, 2)



        # 1. 位置损失

        loss_pos = self._position_loss(pred_coords, target_coords)



        # 2. 修复：相对间距损失（而非绝对间距）

        loss_spacing = self._relative_spacing_loss(pred_coords, target_coords)



        # 3. 新增：形状保持损失

        loss_shape = self._shape_preservation_loss(pred_coords, target_coords)



        # 4. 新增：预测平滑性损失

        loss_smooth = self._smoothness_loss(pred_coords)



        # 计算总损失

        if self.use_adaptive_weights:

            # 自适应权重：根据损失大小动态调整

            weights = self._compute_adaptive_weights(loss_pos, loss_spacing, loss_shape, loss_smooth)

            total_loss = (weights[0] * loss_pos +

                         weights[1] * loss_spacing +

                         weights[2] * loss_shape +

                         weights[3] * loss_smooth)

        else:

            total_loss = (self.w_pos * loss_pos +

                         self.w_spacing * loss_spacing +

                         self.w_shape * loss_shape +

                         self.w_smooth * loss_smooth)



        # 损失组件

        loss_dict = {

            'loss_pos': loss_pos.item(),

            'loss_spacing': loss_spacing.item(),

            'loss_shape': loss_shape.item(),

            'loss_smooth': loss_smooth.item()

        }



        return total_loss, loss_dict



    def _position_loss(self, pred_coords, target_coords):

        """位置损失：基础的坐标误差"""

        if self.loss_type == 'mse':

            return F.mse_loss(pred_coords, target_coords)

        elif self.loss_type == 'mae':

            return F.l1_loss(pred_coords, target_coords)

        elif self.loss_type == 'huber':

            return F.huber_loss(pred_coords, target_coords, delta=1.0)

        else:

            return F.mse_loss(pred_coords, target_coords)



    def _relative_spacing_loss(self, pred_coords, target_coords):

        """修复：相对间距损失 - 保持阵元间的相对距离关系"""

        # 计算相邻阵元间的向量差

        pred_diffs = pred_coords[:, 1:] - pred_coords[:, :-1]  # [B, N-1, 2]

        target_diffs = target_coords[:, 1:] - target_coords[:, :-1]  # [B, N-1, 2]



        # 相对差异损失

        return F.mse_loss(pred_diffs, target_diffs)



    def _shape_preservation_loss(self, pred_coords, target_coords):

        """新增：形状保持损失 - 保持阵列的整体几何形状"""

        batch_size = pred_coords.size(0)



        # 计算阵元间距离矩阵

        pred_dists = torch.cdist(pred_coords, pred_coords)  # [B, N, N]

        target_dists = torch.cdist(target_coords, target_coords)  # [B, N, N]



        # 只考虑上三角矩阵（避免重复计算）

        mask = torch.triu(torch.ones(self.num_elements, self.num_elements), diagonal=1).bool()

        mask = mask.unsqueeze(0).expand(batch_size, -1, -1).to(pred_coords.device)



        pred_dists_masked = pred_dists[mask]

        target_dists_masked = target_dists[mask]



        return F.mse_loss(pred_dists_masked, target_dists_masked)



    def _smoothness_loss(self, pred_coords):

        """新增：平滑性损失 - 防止预测坐标出现突变"""

        # 计算相邻阵元坐标的二阶差分

        if self.num_elements < 3:

            return torch.tensor(0.0, device=pred_coords.device)



        # 二阶差分：(x[i+1] - x[i]) - (x[i] - x[i-1])

        second_diff = pred_coords[:, 2:] - 2 * pred_coords[:, 1:-1] + pred_coords[:, :-2]



        # L2正则化

        return torch.mean(second_diff ** 2)



    def _compute_adaptive_weights(self, loss_pos, loss_spacing, loss_shape, loss_smooth):

        """计算自适应权重"""

        losses = torch.stack([loss_pos, loss_spacing, loss_shape, loss_smooth])



        # 基于损失相对大小的权重调整

        loss_ratios = losses / (torch.mean(losses) + 1e-8)

        adaptive_weights = torch.softmax(-loss_ratios, dim=0)  # 损失越大权重越小



        # 与预设权重结合

        base_weights = torch.tensor([self.w_pos, self.w_spacing, self.w_shape, self.w_smooth])

        final_weights = 0.7 * base_weights + 0.3 * adaptive_weights.cpu()



        return final_weights





class MultiScaleLoss(nn.Module):

    """多尺度损失函数：在不同空间尺度上评估预测质量"""



    def __init__(self, config):

        super().__init__()

        self.num_elements = config['data']['num_elements']

        self.scales = config['loss'].get('scales', [1.0, 0.5, 0.25])  # 不同尺度



        # 为每个尺度创建损失函数

        self.scale_losses = nn.ModuleList([

            SonarLoss(config) for _ in self.scales

        ])



    def forward(self, pred, target):

        total_loss = 0

        loss_dict = {}



        batch_size = pred.size(0)

        pred_coords = pred.reshape(batch_size, self.num_elements, 2)

        target_coords = target.reshape(batch_size, self.num_elements, 2)



        for i, (scale, loss_fn) in enumerate(zip(self.scales, self.scale_losses)):

            # 缩放坐标

            scaled_pred = pred_coords * scale

            scaled_target = target_coords * scale



            # 计算该尺度下的损失

            scale_loss, scale_dict = loss_fn(

                scaled_pred.reshape(batch_size, -1),

                scaled_target.reshape(batch_size, -1)

            )



            total_loss += scale_loss



            # 记录各尺度损失

            for k, v in scale_dict.items():

                loss_dict[f'scale_{i}_{k}'] = v



        return total_loss, loss_dict





class RobustSonarLoss(nn.Module):

    """鲁棒声呐损失：对异常值不敏感"""



    def __init__(self, config):

        super().__init__()

        self.base_loss = SonarLoss(config)

        self.use_robust = config['loss'].get('use_robust', False)

        self.robust_threshold = config['loss'].get('robust_threshold', 2.0)



    def forward(self, pred, target):

        if not self.use_robust:

            return self.base_loss(pred, target)



        # 计算基础损失

        total_loss, loss_dict = self.base_loss(pred, target)



        # 对异常样本进行软权重处理

        batch_size = pred.size(0)

        pred_coords = pred.reshape(batch_size, self.num_elements, 2)

        target_coords = target.reshape(batch_size, self.num_elements, 2)



        # 计算每个样本的误差

        sample_errors = torch.mean((pred_coords - target_coords) ** 2, dim=[1, 2])  # [B]



        # 基于误差大小计算权重（误差越大权重越小）

        error_threshold = torch.quantile(sample_errors, 0.75)  # 75分位数作为阈值

        weights = torch.where(

            sample_errors > error_threshold,

            torch.exp(-(sample_errors - error_threshold) / self.robust_threshold),

            torch.ones_like(sample_errors)

        )



        # 加权损失

        weighted_loss = total_loss * torch.mean(weights)



        loss_dict['robust_weight'] = torch.mean(weights).item()



        return weighted_loss, loss_dict





def get_loss_function(config):

    """损失函数工厂函数"""

    loss_type = config['loss'].get('loss_class', 'SonarLoss')



    if loss_type == 'SonarLoss':

        return SonarLoss(config)

    elif loss_type == 'MultiScaleLoss':

        return MultiScaleLoss(config)

    elif loss_type == 'RobustSonarLoss':

        return RobustSonarLoss(config)

    else:

        raise ValueError(f"未知的损失函数类型: {loss_type}")





class LossScheduler:

    """损失权重调度器：在训练过程中动态调整各损失组件的权重"""



    def __init__(self, config):

        self.schedule_type = config['loss'].get('weight_schedule', 'constant')

        self.total_epochs = config['train']['epochs']



    def get_weights(self, epoch):

        """根据训练进度返回当前权重"""

        progress = epoch / self.total_epochs



        if self.schedule_type == 'constant':

            return None  # 使用默认权重



        elif self.schedule_type == 'annealing':

            # 逐渐降低形状损失权重，增加位置损失权重

            w_pos = 1.0 + 0.5 * progress

            w_spacing = 0.1 * (1 - 0.5 * progress)

            w_shape = 0.05 * (1 - 0.8 * progress)

            w_smooth = 0.01 * (1 - 0.3 * progress)



            return {

                'weight_position': w_pos,

                'weight_spacing': w_spacing,

                'weight_shape': w_shape,

                'weight_smooth': w_smooth

            }



        return None