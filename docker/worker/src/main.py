from task import Task
from job import Job

if __name__ == '__main__':
    from kubernetes.config import load_incluster_config

    with open('test.json', 'r') as file:
        config = file.read()

    load_incluster_config()

    task = Task('test', config)

    job = Job('test')
