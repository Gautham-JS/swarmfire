import argparse

from config.Config import EnvConfig
from train.trxl_train_single_agent import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CleanRL TrXL PPO - FireScout (parallel)")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = EnvConfig()
    train(cfg, checkpoint_path=args.checkpoint)