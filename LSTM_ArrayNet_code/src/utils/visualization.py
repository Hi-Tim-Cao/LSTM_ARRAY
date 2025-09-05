"""结果可视化工具"""

import matplotlib.pyplot as plt

import numpy as np





def plot_position_comparison(pred, target, save_path=None):

    """

    绘制预测坐标与真实坐标对比图

    :param pred: 预测坐标 [num_elements, 2]

    :param target: 真实坐标 [num_elements, 2]

    :param save_path: 保存路径（None则显示）

    """

    plt.figure(figsize=(8, 6))



    # 绘制真实坐标和预测坐标

    plt.scatter(target[:, 0], target[:, 1], c='blue', label='真实坐标', marker='o', s=50)

    plt.scatter(pred[:, 0], pred[:, 1], c='red', label='预测坐标', marker='x', s=50)



    # 连接对应阵元（显示配对关系）

    for p, t in zip(pred, target):

        plt.plot([p[0], t[0]], [p[1], t[1]], 'gray', linestyle='--', alpha=0.5)



    plt.xlabel('X坐标 (m)', fontsize=12)

    plt.ylabel('Y坐标 (m)', fontsize=12)

    plt.title('阵列位置估计对比', fontsize=14)

    plt.legend(fontsize=10)

    plt.grid(alpha=0.3)



    if save_path:

        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.close()

    else:

        plt.show()