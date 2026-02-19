import json
import struct
import redis
import numpy as np



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
    
