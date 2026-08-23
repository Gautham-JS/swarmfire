from eval.eval_trxl import run_evaluation
import argparse
from config.Config import EnvConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to the trained "
            "PyTorch checkpoint."
        )
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help=(
            "Number of evaluation "
            "episodes."
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # =========================================================================
    # IMPORTANT:
    #
    # This should be the same EnvConfig used during training.
    #
    # Replace this with however your project constructs EnvConfig.
    # =========================================================================
    cfg = EnvConfig()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        n_episodes=args.episodes
    )