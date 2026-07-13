from dataclasses import dataclass, field

MAXINT = 99999999


@dataclass
class VideoWriterConfig:
    is_enabled          : bool          = False
    run_id              : str           = "RLVideo"
    sample_interval     : int           = MAXINT
    save_interval       : int           = MAXINT
    base_path           : str           = "./vids/"
    fps                 : int           = 30


@dataclass
class EnvConfig:
    run_id:           str   = None

    # Environment
    world_size:       tuple = (512, 512)
    n_agents:         int   = 1
    iter_limit:       int   = 1028
    seed:             int   = None
    n_envs:           int   = 8          # parallel environments

    # TrXL
    features_dim:     int   = 256
    memory_len:       int   = 128
    n_layers:         int   = 4
    n_heads:          int   = 4
    d_ff_multiplier:  int   = 2
    dropout:          float = 0.1

    # PPO
    total_timesteps:  int   = 2_000_000
    n_steps:          int   = 512        # steps per env per rollout
                                         # total transitions = n_steps * n_envs. 
    batch_size:       int   = 256        # minibatch size for PPO update
    n_epochs:         int   = 10
    learning_rate:    float = 1e-4
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_coef:        float = 0.2
    ent_coef:         float = 0.0001
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.3
    target_kl:        float = 0.03

    # Env weights
    phase_weights:    dict  = field(default_factory=lambda: {
        "exploration":          1.0,
        "exploration_tracking": 0.05,
        "fire_discovery":       18.8,
        "fire_tracking":        10.5,
        "risk":                 1.5,
    })

    # Checkpointing
    checkpoint_freq:  int   = 50_000
    checkpoint_dir:   str   = "./checkpoints"
    best_model_dir:   str   = "./best_model"

    # Evaluation
    eval_freq:        int   = 50_000
    n_eval_episodes:  int   = 5

    # WandB
    wandb_project:    str   = "thesis-drl-trxl"
    wandb_api_key:    str   = "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"

