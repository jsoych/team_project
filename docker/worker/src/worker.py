# worker_list.py
import os
import time

from task import Task

class Worker:
    def __init__(self, status = 'not-assigned'):
        self.status = status
        self.task: Task = None
        pass

    def getStatus(self):
        return self.status
    
    def setStatus(self, status):
        self.status = status
        return
    
    def updateStatus(self):
        if (worker.status != 'working'):
            return
        
        if (self.task.getStatus() != 'running'):
            worker.status = 'not-working'

        return

    def run(self, task):
        ''' Runs the task '''
        # check worker status
        if (self.status == 'not-assigned'):
            old_task = None
            self.bind(task)
        elif (self.status == 'not-working'):
            old_task = self.unbind()
            self.bind(task)
            return old_task
        else:
            return None
        
        # run task
        self.task.run()
        self.status = 'working'
        return old_task
    
    def bind(self, task):
        ''' Binds the task to the worker. '''
        self.task = task
        return
    
    def unbind(self, task) -> Task:
        ''' Unbinds the task from the worker and returns it. '''
        old_task = self.task
        return old_task

    def getTaskStatus(self):
        ''' Returns the status of the worker's task. '''
        if (self.status == 'not-assigned'):
            return None
        
        return self.task.getStatus()


if __name__ == '__main__':
    from redis import Redis

    r = Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'))
    worker = Worker()
    while True:
        # update worker status
        worker.updateStatus()

        # check worker status
        if (worker.getStatus() == 'working'):
            time.sleep(1)
            continue

        # get task
        _, raw = r.brpop(["experiments:ready"], timeout=1)

        if (task := Task().decode(raw)):
            # handle decoding error
            r.rpush('experiments:not-ready', raw)

        # run task
        if (old_task := worker.run(task) == None):
            continue

        # check old task status
        if (old_task.getStatus() == 'incomplete'):
            r.lpush('experiments:incomplete', old_task.encode())
        elif (old_task.getStatus() == 'completed'):
            r.lpush('experiments:completed', old_task.encode())
