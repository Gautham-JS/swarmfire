import os

class ServerConfig:
    HOST            = os.getenv("WF_HOST",       "0.0.0.0")
    PORT            = int(os.getenv("WF_PORT",   "8000"))
    MODEL_PATH      = os.getenv("WF_MODEL",      "models/unet_wildfire.pth")
    SAVE_FRAMES     = os.getenv("WF_SAVE",       "false").lower() == "true"
    SAVE_DIR        = os.getenv("WF_SAVE_DIR",   "./received_frames/")
    MAX_CLIENTS     = int(os.getenv("WF_MAX_CL", "10"))
    DEVICE          = os.getenv("WF_DEVICE",     "cuda")
    FIRE_THRESHOLD  = float(os.getenv("WF_FIRE_THRESH", "0.01"))
    LOG_LEVEL       = os.getenv("WF_LOG",        "INFO")

cfg = ServerConfig()