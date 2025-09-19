import json
import time

from kubernetes import client, config

config.load_config()
v1 = client.CoreV1Api()

class Task:
    def __init__(self, data=None):
        self.cm = v1.create_namespaced_config_map(
            namespace='team-project',
            body=client.V1ConfigMap(
                api_version='v1',
                metadata={
                    'name': 'test-cm',
                    'labels': {
                        'stage': 'dev',
                        'delete': 'true'
                    }
                },
                data=data
            )
        )
        return

    def getStatus(self):
        ''' Returns the status of the task. '''
        return self.status
    
    def setStatus(self, status):
        ''' Sets the status of the task. '''
        self.status = status
        return

    def decode(self, raw):
        ''' Decodes the raw task configuration and returns True. Otherwise,
            returns False. '''
        # unmarshal the task
        if (obj := json.loads(raw) == None):
            return False
        
        # check if the object has the needed keys
        for k in ['id']:
            if (k not in obj.keys()):
                return False
        
        # decode obj
        self.id = obj['id']
        return True
        
    def encode(self):
        ''' Encodes the task into its raw configuration. '''
        obj = {
            'id': self.id,
            'status': self.status
        }
        return json.dumps(obj).encode()
    
    def apply(self, data=None):
        ''' Apply changes to task. '''
        v1.patch_namespaced_config_map(
            name=self.cm.metadata.name,
            namespace=self.cm.metadata.namespace,
            body={'data': data}
        )
        return

if __name__ == '__main__':
    import os
    with open(os.getenv('EXPERIMENT_CONFIG'), 'r') as f:
        data = f.read(-1)
    
    task = Task({'experiment.yaml': data})
