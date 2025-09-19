import os
import json

from redis import Redis

queue = Redis(
    host=os.getenv('REDIS_HOST', "redis-master"), 
    password=os.getenv('REDIS_PASSWORD')
)

def push_experiment(id, config):
    experiment = {'id': id, 'config': config}
    queue.rpush('experiments:queue', json.dumps(experiment))


for id in range(10):
    push_experiment(id, {'name': 'test', 'model': 'svm'})
