"""配置文件加载与合并"""

import yaml





def load_config(config_path):

    """加载主配置文件并合并基础配置"""

    # 加载基础配置

    with open('configs/base.yaml', 'r') as f:

        base_config = yaml.safe_load(f)



    # 加载模型专用配置

    with open(config_path, 'r') as f:

        model_config = yaml.safe_load(f)



    # 合并配置（模型配置覆盖基础配置）

    return _merge_configs(base_config, model_config)





def _merge_configs(base, update):

    """递归合并两个配置字典"""

    merged = base.copy()

    for k, v in update.items():

        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):

            merged[k] = _merge_configs(merged[k], v)

        else:

            merged[k] = v

    return merged