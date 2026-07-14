import redis
import hashlib
from typing import Optional

redis_client = redis.Redis(
    host="localhost",  # if you are using docker, use the name of the container instead of localhost
    port=6379,
    decode_responses=True,
)

"""
Cache : key value ttl
"""

class ResponseCache:
    def __init__(self, redis_client: redis.Redis ,ttl=3600):
        self.redis_client = redis_client
        self.ttl = ttl

    def make_key(self, query: str) -> str:
        normalized = " ".join(query.lower().split())        
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"response_cache:{digest}"

    def set(self, query: str, value: str, expiration: int = None):
        key = self.make_key(query)
        self.redis_client.set(key, value, ex=self.ttl)

    def get(self, query: str) -> Optional[str]:
        key = self.make_key(query)
        return self.redis_client.get(key)

    def delete(self, query: str):
        key = self.make_key(query)
        self.redis_client.delete(key)
