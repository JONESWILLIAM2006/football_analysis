# Mock Redis Implementation for Development
import json
import time
from collections import defaultdict, deque

class MockRedis:
    """Mock Redis implementation for development without Redis server"""
    
    def __init__(self, host='localhost', port=6379, db=0, decode_responses=True):
        self.data = {}
        self.lists = defaultdict(deque)
        self.hashes = defaultdict(dict)
        self.connected = True
        
    def ping(self):
        return True
    
    def set(self, key, value):
        self.data[key] = value
        return True
    
    def get(self, key):
        return self.data.get(key)
    
    def delete(self, *keys):
        count = 0
        for key in keys:
            if key in self.data:
                del self.data[key]
                count += 1
            if key in self.lists:
                del self.lists[key]
                count += 1
            if key in self.hashes:
                del self.hashes[key]
                count += 1
        return count
    
    def lpush(self, key, *values):
        for value in values:
            self.lists[key].appendleft(value)
        return len(self.lists[key])
    
    def rpop(self, key):
        if key in self.lists and self.lists[key]:
            return self.lists[key].pop()
        return None
    
    def llen(self, key):
        return len(self.lists[key])
    
    def hset(self, key, mapping=None, **kwargs):
        if mapping:
            self.hashes[key].update(mapping)
        if kwargs:
            self.hashes[key].update(kwargs)
        return len(self.hashes[key])
    
    def hget(self, key, field):
        return self.hashes[key].get(field)
    
    def hgetall(self, key):
        return dict(self.hashes[key])

# Create global mock Redis instance
mock_redis_instance = MockRedis()

def get_redis_client():
    """Get Redis client - returns mock if real Redis unavailable"""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        client.ping()  # Test connection
        return client
    except:
        print("⚠️ Using Mock Redis (Redis server not available)")
        return mock_redis_instance