from kubernetes.client import CoreV1Api, V1ConfigMap
from kubernetes.client.exceptions import ApiException

class Task:
    def __init__(self, name, config, logger=None):
        self.name = name
        self.config = config
        self.__logger = logger
        try:
            if (self.__logger is not None):
                self.__logger.debug('creating configmap')
            self.cm = CoreV1Api().create_namespaced_config_map(
                namespace='team-project',
                body=V1ConfigMap(
                    api_version='v1',
                    metadata={
                        'name': name,
                        'labels': {
                            'stage': 'dev',
                            'delete': 'true'
                        }
                    },
                    data={'experiment.json': config}
                )
            )
        
        except ApiException as err:
            if (self.__logger is not None):
                self.__logger.debug(f'status: {err.status} reason: {err.reason}')
            self.cm = None
        return
    
    def get_status(self):
        ''' Gets the status of the configmap. '''
        try:
            resp = CoreV1Api().get_api_resources()
        except ApiException as err:
            print(err)
            resp = None

        return resp

    
    def delete(self):
        try:
            if(self.__logger is not None):
                self.__logger.debug('deleting configmap')
            CoreV1Api().delete_namespaced_config_map(
                name=self.name, namespace='team-project'
            )
        except ApiException as err:
            if (self.__logger is not None):
                self.__logger.debug(f'status: {err.status} reason: {err.reason}')
            else:
                print(err)
    
    def __del__(self):
        try:
            self.delete()
        except Exception:
            pass
    
if __name__ == '__main__':
    import yaml

    from logging import getLogger
    from logging.config import dictConfig
    from time import sleep
    from kubernetes.config import load_config

    with open('/home/jsoychak/cs/team_project/docker/worker/assets/logger.yaml', 'rb') as f:
        config = yaml.safe_load(f)
        dictConfig(config)
    
    logger = getLogger(__name__)

    load_config()

    task = Task('test', 'configuration', logger)
    print(task.get_status())
    sleep(5)

    task.delete()
    sleep(5)

    task = Task('test', 'configuration', logger)
    task = None
    sleep(5)
