import yaml

from kubernetes.client import BatchV1Api
from task import Task

def make_template():
    ''' Makes a job template and returns it. '''
    with open('job-template.yaml', 'rb') as file:
        template = yaml.safe_load(file)
    return template

def insert_name(template, name):
    ''' Inserts the name into the job template. '''
    template['metadata']['name'] = name
    return

def insert_configmap(template, name):
    ''' Inserts the config map into the job template. '''
    volumes = template['spec']['template']['spec']['volumes']
    volumes.append({'name': 'configs', 'configMap': {'name': name}})
    return

def create_job(name):
    ''' Creates a new kubernetes job and returns api response. '''

    # create template
    template = make_template()

    # insert values
    insert_name(template, name)
    insert_configmap(template, name)
    
    # create job
    resp = BatchV1Api().create_namespaced_job(
        namespace='team-project',
        body=template
    )
    return resp

class Job(object):
    
    def __init__(self, name, tasks=None, logger=None):
        '''
        Job Class wraps the k8s job resource and its API.

        paramaters:
        - name: string
        - status: string
        - tasks: None | list
        '''
        self.name = name
        self.status = 'not-ready'
        self.tasks = tasks
        self.__logger = logger
        return
    
    def __update_status(self):
        # when is the status ready?
        # - its tasks are created
        # - its name is available
        if (self.task == None):
            self.status = 'ready'

        for task in self.tasks:
            try:
                status = task.get_status()
            except Exception as err:
                if (self.__logger is not None):
                    self.__logger.debug(err)
                else:
                    print(err)
        return
    
    def get_status(self):
        self.__update_status()
        return self.status
    
    def add_task(self, task):
        ''' Adds the task to the list of job tasks. '''
        self.tasks.append(task)
        return
    
    def run(self):
        ''' Creates the job tasks. '''
        return

if __name__ == '__main__':
    import urllib3
    import json

    from kubernetes.config import load_config
    from kubernetes.client.api_client import ApiClient
    from kubernetes.dynamic.exceptions import NotFoundError

    load_config()

    job = Job('test')

    client = ApiClient()

    resource_version = None
    try:
        raw, status, headers = client.call_api('/api/v1/namespaces/team-project/pods', 'GET')
        print(json.loads(raw))
    except NotFoundError:
        print('Job does not exist')
    '''
    try:
        for event in api.watch(namespace='team-project', resource_version=resource_version):
            resource_version = event['object'].metadata.resourceVersion

            print(f'Event: {resource_version} {event['type']} {event['object'].metadata.name}')
    
    except ApiException as err:
        if (err.status == 410):
            print('ERROR: The requested resource version is no longer available')
        else:
            raise
    
    except urllib3.exceptions.ProtocolError:
        print('Lost connection to the k8s API server. Reconnecting...')
    '''
