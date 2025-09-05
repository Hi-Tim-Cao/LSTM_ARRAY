"""LSTM声呐阵列位置估计工程入口"""

import argparse

from src.utils.config import load_config



def main():

    parser = argparse.ArgumentParser(description="声呐阵列位置估计LSTM模型")

    parser.add_argument("--task", type=str, required=True, choices=["train", "predict"],

                        help="任务类型：train（训练）或predict（推理）")

    parser.add_argument("--config", type=str, default="configs/lstm.yaml",

                        help="配置文件路径")

    parser.add_argument("--model_path", type=str,

                        help="推理时需指定模型权重路径（如experiments/.../best_model.pth）")

    args = parser.parse_args()



    # 验证参数

    if args.task == "predict" and not args.model_path:

        raise ValueError("推理任务必须通过--model_path指定模型权重路径")



    # 执行任务

    if args.task == "train":

        from src.train.trainer import start_training

        start_training(args.config)

    else:

        from src.inference.predictor import start_inference

        start_inference(args.config, args.model_path)



if __name__ == "__main__":

    main()