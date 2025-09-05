"""模型推理与评估 - 修复版本"""

import os

import json

import torch

import numpy as np

from src.models.lstm_attention import LSTMAttention

from src.utils.metrics import calc_rmse, calc_hausdorff

from src.utils.visualization import plot_position_comparison





class Predictor:

    def __init__(self, config, model_path):

        self.config = config

        self.device = torch.device(config['train']['device'])

        self.model_path = model_path

        self.num_elements = config['data']['num_elements']



        # 修复：添加异常处理

        try:

            # 加载模型

            self.model = LSTMAttention(config).to(self.device)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

            self.model.eval()

        except Exception as e:

            raise RuntimeError(f"模型加载失败: {e}")



        # 结果保存路径

        self.exp_dir = os.path.dirname(os.path.dirname(model_path))

        self.results_dir = os.path.join(self.exp_dir, 'results')

        os.makedirs(self.results_dir, exist_ok=True)



    def predict(self, test_loader, preprocessor):

        """批量预测并评估"""

        all_preds = []

        all_labels = []



        print("开始推理...")

        with torch.no_grad():

            for batch_idx, (features, labels) in enumerate(test_loader):

                # 修复：正确调用预处理方法

                features_scaled, labels_scaled = self._preprocess_batch(

                    features, labels, preprocessor

                )



                # 模型预测

                features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)

                outputs = self.model(features_tensor)



                # 保存结果

                all_preds.append(outputs.cpu().numpy())

                all_labels.append(labels_scaled)



                if batch_idx % 10 == 0:

                    print(f"已处理 {batch_idx+1}/{len(test_loader)} 个批次")



        # 修复：正确处理批次结果

        all_preds = np.vstack(all_preds)  # [N_samples, num_elements*2]

        all_labels = np.vstack(all_labels)  # [N_samples, num_elements*2]



        # 反归一化并重塑为正确形状

        all_preds_orig = preprocessor.inverse_transform_labels(all_preds)  # 恢复原始坐标

        all_labels_orig = preprocessor.inverse_transform_labels(all_labels)



        # 重塑为 [N_samples, num_elements, 2] 格式用于指标计算

        all_preds_reshaped = all_preds_orig.reshape(-1, self.num_elements, 2)

        all_labels_reshaped = all_labels_orig.reshape(-1, self.num_elements, 2)



        # 计算评估指标

        metrics = {

            'rmse': calc_rmse(all_preds_reshaped, all_labels_reshaped),

            'hausdorff': calc_hausdorff(all_preds_reshaped, all_labels_reshaped),

            'mae': np.mean(np.abs(all_preds_reshaped - all_labels_reshaped)),  # 添加MAE指标

            'num_samples': len(all_preds_reshaped)

        }



        # 保存结果和可视化

        self._save_results(metrics, all_preds_reshaped, all_labels_reshaped)

        return metrics



    def _preprocess_batch(self, features, labels, preprocessor):

        """修复：正确的批处理预处理方法"""

        # features: [B, N, T, F]（numpy数组）

        # labels: [B, N, 2]（numpy数组）

        batch_size = len(features)

        features_scaled = []

        labels_scaled = []



        for i in range(batch_size):

            # 对每个样本进行预处理

            f_scaled, l_scaled = preprocessor.transform(features[i], labels[i])

            features_scaled.append(f_scaled)



            # 修复：正确处理标签维度

            # l_scaled 应该是 [num_elements*2] 的形状

            if l_scaled.ndim == 1:

                l_scaled = l_scaled.reshape(-1)  # 确保是1D

            labels_scaled.append(l_scaled)



        return np.array(features_scaled), np.array(labels_scaled)



    def predict_single(self, features, preprocessor):

        """单样本推理方法"""

        self.model.eval()

        with torch.no_grad():

            # 预处理

            features_scaled, _ = preprocessor.transform(features, np.zeros((self.num_elements, 2)))



            # 添加batch维度

            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)



            # 预测

            output = self.model(features_tensor)

            pred_scaled = output.cpu().numpy()[0]  # 移除batch维度



            # 反归一化

            pred_orig = preprocessor.inverse_transform_labels(pred_scaled.reshape(1, -1))

            return pred_orig.reshape(self.num_elements, 2)



    def _save_results(self, metrics, preds, labels):

        """保存评估结果和可视化对比图"""

        # 保存指标

        metrics_file = os.path.join(self.results_dir, 'metrics.json')

        with open(metrics_file, 'w') as f:

            # 修复：确保所有值都可序列化

            serializable_metrics = {}

            for k, v in metrics.items():

                if isinstance(v, np.floating):

                    serializable_metrics[k] = float(v)

                elif isinstance(v, np.integer):

                    serializable_metrics[k] = int(v)

                else:

                    serializable_metrics[k] = v

            json.dump(serializable_metrics, f, indent=4)



        # 保存详细结果（可选）

        results_file = os.path.join(self.results_dir, 'detailed_results.npz')

        np.savez(results_file, predictions=preds, ground_truth=labels)



        # 随机选择样本可视化

        num_vis = min(10, len(preds))

        indices = np.random.choice(len(preds), num_vis, replace=False)



        for i, idx in enumerate(indices):

            try:

                plot_position_comparison(

                    preds[idx], labels[idx],

                    save_path=os.path.join(self.results_dir, f'comparison_{i}.png')

                )

            except Exception as e:

                print(f"保存可视化图表 {i} 时出错: {e}")



        print(f"推理结果保存至：{self.results_dir}")

        print(f"测试集RMSE：{metrics['rmse']:.4f}m")

        print(f"测试集MAE：{metrics['mae']:.4f}m")

        print(f"Hausdorff距离：{metrics['hausdorff']:.4f}m")

        print(f"测试样本数：{metrics['num_samples']}")



    def evaluate_model_performance(self, test_loader, preprocessor):

        """详细的模型性能分析"""

        metrics = self.predict(test_loader, preprocessor)



        # 添加更多分析

        print("\n=== 详细性能分析 ===")

        print(f"平均定位误差：{metrics['rmse']:.4f}m")

        print(f"最大形状误差：{metrics['hausdorff']:.4f}m")



        # 可以添加更多分析，如：

        # - 每个阵元的单独精度

        # - 误差分布统计

        # - 不同条件下的性能差异



        return metrics





def start_inference(config_path, model_path):

    """推理入口函数"""

    from src.utils.config import load_config

    from src.data.dataloader_builder import build_dataloaders



    try:

        # 加载配置

        config = load_config(config_path)



        # 构建测试数据加载器

        test_loader, preprocessor = build_dataloaders(config, is_train=False)



        # 启动推理

        predictor = Predictor(config, model_path)

        metrics = predictor.evaluate_model_performance(test_loader, preprocessor)



        return metrics



    except Exception as e:

        print(f"推理过程出错: {e}")

        raise