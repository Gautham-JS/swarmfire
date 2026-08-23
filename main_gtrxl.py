import argparse
import logging

from config.Config import EnvConfig
from train.trxl_train_single_agent import train
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanRL TrXL PPO - FireScout (parallel)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, required=True)
    parser.add_argument("-g", "--no_gating", type=bool, default=False)    
    parser.add_argument("-s", "--no_spatial_bias", type=bool, default=False)
    parser.add_argument("-H", "--no_hyperconnect", type=bool, default=False)
    parser.add_argument("-m", "--memory_len", type=int, default=128)
    parser.add_argument("-d", "--device", type=str, default="cuda:0")

    args = parser.parse_args()

    cfg = EnvConfig()
    cfg.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg.checkpoint_dir = f"./checkpoints_{args.name}"

    cfg.is_gating = not args.no_gating
    cfg.is_spatial_bias = not args.no_spatial_bias
    cfg.is_hyperconnect = not args.no_hyperconnect
    cfg.memory_len = args.memory_len
    # cfg.features_dim = 512
    cfg.iter_limit = 1024

    logging.info(f"--> Using device: {cfg.device}")
    logging.info(f"--> Run name: {cfg.run_name}")
    logging.info(f"--> Gating: {cfg.is_gating} | Spatial Bias: {cfg.is_spatial_bias} | Hyperconnect: {cfg.is_hyperconnect}")
    logging.info(f"--> Memory length: {cfg.memory_len}")

    cfg.run_name = args.name
    train(cfg, checkpoint_path=args.checkpoint)