"""评估指标计算"""

import numpy as np

from scipy.spatial.distance import cdist



def calc_rmse(pred, target):

    """

    计算均方根误差（Root Mean Square Error）

    :param pred: 预测坐标，形状 [N_samples, num_elements, 2]

    :param target: 真实坐标，形状 [N_samples, num_elements, 2]

    :return: 平均RMSE（米）

    """

    return np.sqrt(np.mean((pred - target)** 2))



def calc_hausdorff(pred, target):

    """

    计算Hausdorff距离（衡量两个点集的整体差异）

    :param pred: 预测坐标，形状 [N_samples, num_elements, 2]

    :param target: 真实坐标，形状 [N_samples, num_elements, 2]

    :return: 平均Hausdorff距离（米）

    """

    distances = []

    for p, t in zip(pred, target):

        # 计算点集间的距离矩阵

        dist_matrix = cdist(p, t)

        # 双向最大最小距离

        h1 = np.max(np.min(dist_matrix, axis=1))  # 从pred到target

        h2 = np.max(np.min(dist_matrix, axis=0))  # 从target到pred

        distances.append(max(h1, h2))

    return np.mean(distances)