#!/usr/bin/env python3
# Redis Connection Test

import redis
import sys

def test_redis_connection():
    """Test Redis connection and basic operations"""
    try:
        # Try to connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test ping
        if r.ping():
            print("✅ Redis connection successful!")
            
            # Test basic operations
            r.set('test_key', 'test_value')
            value = r.get('test_key')
            
            if value == 'test_value':
                print("✅ Redis read/write operations working")
                
                # Clean up
                r.delete('test_key')
                
                # Test job queue operations
                r.lpush('test_queue', 'job1', 'job2', 'job3')
                queue_length = r.llen('test_queue')
                print(f"✅ Queue operations working (length: {queue_length})")
                
                # Clean up queue
                r.delete('test_queue')
                
                return True
            else:
                print("❌ Redis read/write operations failed")
                return False
        else:
            print("❌ Redis ping failed")
            return False
            
    except redis.ConnectionError:
        print("❌ Could not connect to Redis")
        print("💡 Try starting Redis:")
        print("   - macOS: brew install redis && brew services start redis")
        print("   - Docker: docker run -d -p 6379:6379 redis:6-alpine")
        print("   - Manual: redis-server")
        return False
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False

if __name__ == "__main__":
    success = test_redis_connection()
    sys.exit(0 if success else 1)