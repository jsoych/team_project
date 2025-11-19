# worker_list.py
import os
import time

from job import Job
from task import Task

class Worker:
    def __init__(self, name):
        self.status = 'not-assigned'
        self.job = Job(name)
        pass

    def getStatus(self):
        return self.status
    
    def setStatus(self, status):
        self.status = status
        return
    
    def updateStatus(self):
        if (worker.status != 'working'):
            return
        
        if (self.job.getStatus() != 'running'):
            worker.status = 'not-working'

        return
    
    def addTask(self, task):
        ''' Adds a task to the job to the worker's job. '''
        self.job.addTask(task)
        return

    def run(self):
        ''' Runs the job and returns the status of the old job. '''
        # check worker status
        if (self.status == 'not-assigned'):
            job_status = None
        elif (self.status == 'not-working'):
            job_status = self.job.getStatus()
        
        self.job.run()
        self.status = 'working'
        return job_status

    def getJobStatus(self):
        ''' Returns the status of the worker's task. '''
        if (self.status == 'not-assigned'):
            return None
        
        return self.job.getStatus()


if __name__ == '__main__':
    from redis import Redis
    from kubernetes.config import load_incluster_config

    r = Redis(host=os.getenv('REDIS_HOST'), port=os.getenv('REDIS_PORT'))
    load_incluster_config()

    worker = Worker('danny')
    
    while True:
        # update worker status
        worker.updateStatus()

        # check worker status
        if (worker.getStatus() == 'working'):
            time.sleep(1)
            continue

        # get task
        reply = r.brpop(["experiments:ready"], timeout=1)

        # check raw
        if (reply == None):
            continue

        # unpack reply
        _, raw = reply

        # create task
        task = Task('danny', str(raw))
        
        # add task to worker
        worker.addTask(task)

        # run job
        old_job_status = worker.run()
