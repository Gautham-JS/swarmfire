import json
import struct
import redis
import numpy as np

import pickle
import zlib
import logging
from typing import Any, Optional

import numpy as np
import redis

log = logging.getLogger(__name__)

_NDARRAY_MARKER = "__ndarray__"


def encode_payload(obj):
    """
    Recursively replace numpy arrays/scalars with plain, numpy-version-
    independent representations so pickle never touches numpy's own
    array-reduce machinery. Safe to call on arbitrarily nested
    dict/list/tuple structures.
    """
    if isinstance(obj, np.ndarray):
        return {
            _NDARRAY_MARKER: True,
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data": np.ascontiguousarray(obj).tobytes(),
        }
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: encode_payload(v) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return ["__tuple__", [encode_payload(v) for v in obj]]
    if isinstance(obj, list):
        return [encode_payload(v) for v in obj]
    return obj


def decode_payload(obj):
    """Inverse of encode_payload()."""
    if isinstance(obj, dict):
        if obj.get(_NDARRAY_MARKER):
            arr = np.frombuffer(obj["data"], dtype=obj["dtype"])
            return arr.reshape(obj["shape"])
        return {k: decode_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) == 2 and obj[0] == "__tuple__" and isinstance(obj[1], list):
            return tuple(decode_payload(v) for v in obj[1])
        return [decode_payload(v) for v in obj]
    return obj

class RedisRenderPublisher:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8090,
        db: int = 0,
        channel_prefix: str = "render",
        compress: bool = True,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        stream_maxlen: int = 5000,
    ):
        """
        ... (same docstring as before) ...
        """
        self.channel_prefix = channel_prefix
        self.compress = compress
        self.stream_key = channel_prefix  # channel_prefix IS the stream key now
        self.stream_maxlen = stream_maxlen
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                socket_timeout=socket_timeout,
            )
            self.client.ping()
            self._available = True
        except redis.exceptions.RedisError as e:
            log.warning(
                f"[RedisRenderPublisher] Could not connect to redis at {host}:{port} -> {e}. "
                f"Render publish calls will be silently skipped."
            )
            self.client = None
            self._available = False

    # ------------------------------------------------------------------
    def _serialize(self, payload: dict) -> bytes:
        encoded = encode_payload(payload)
        blob = pickle.dumps(encoded, protocol=pickle.HIGHEST_PROTOCOL)
        if self.compress:
            blob = zlib.compress(blob, level=3)
        return blob

    def _publish(self, env_id: str, payload: dict) -> None:
        if not self._available:
            return
        try:
            data = self._serialize(payload)
            self.client.xadd(
                self.stream_key,
                {"env_id": env_id, "data": data},
                maxlen=self.stream_maxlen,
                approximate=True,
            )
        except redis.exceptions.RedisError as e:
            log.warning(f"[RedisRenderPublisher] publish failed: {e}")

    # ------------------------------------------------------------------
    # Public API used by SingleAgentEnv -- unchanged signatures.
    # ------------------------------------------------------------------
    def publish_episode_start(
        self,
        env_id: str,
        episode_count: int,
        world_size,
        vp_size: int,
        render_mode: str,
        video_config: Any,
    ):
        self._publish(env_id, {
            "type": "reset",
            "env_id": env_id,
            "episode_count": episode_count,
            "world_size": world_size,
            "vp_size": vp_size,
            "render_mode": render_mode,
            "video_config": {
                "is_enabled": getattr(video_config, "is_enabled", False),
                "base_path": getattr(video_config, "base_path", None),
                "fps": getattr(video_config, "fps", 30),
                "sample_interval": getattr(video_config, "sample_interval", 1),
            } if video_config is not None else None,
        })

    def publish_frame(self, env_id: str, **kwargs):
        payload = {"type": "frame", "env_id": env_id}
        payload.update(kwargs)
        self._publish(env_id, payload)

    def publish_close(self, env_id: str):
        self._publish(env_id, {"type": "close", "env_id": env_id})

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except redis.exceptions.RedisError:
                pass


def create_redis_client(redis_host='localhost', redis_port=6379):
   return redis.Redis(host=redis_host, port=redis_port, decode_responses=True)


def numpy_to_redis(r, key, array):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    data = json.dumps(array.tolist())
    r.set(key, data)
    return

def numpy_from_redis(r, key):
    """Retrieve Numpy array from Redis key 'n'"""
    data = r.get(key)
    arr = np.array(json.loads(data))
    return arr




class RedisClient:
    def __init__(self, redis_host = "localhost", redis_port=6379):
      self.r = create_redis_client(redis_host, redis_port)

    def set_numpy(self, key, array):
       return numpy_to_redis(self.r, key, array)

    def get_numpy(self, key):
       return numpy_from_redis(self.r, key)

    def clear_keys(self, keys):
       self.r.delete(*keys)