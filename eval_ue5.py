from eval.eval_trxl_ue5 import run_evaluation
import argparse
from config.Config import EnvConfig
from comms.web_sockets.server import handle_client, WSCommsHandler, start_eval_server
import threading
import asyncio

import logging

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
    cfg.wandb_project = "thesis-drl-trxl-eval"

    t = threading.Thread(target=run_evaluation, args=(
        args.checkpoint,
        cfg,
        args.episodes
    ))
    t.start()
    
    asyncio.run(start_eval_server("0.0.0.0", 8090))