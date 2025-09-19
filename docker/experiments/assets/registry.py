import json
import pickle
import socket

BUFFSIZE = 1024

def save_model(model, con :socket.socket, experiment_name):
    ''' Saves the model to the registry and returns the registry model id. '''
    # create save model request
    obj = json.dumps({'experiment_name': experiment_name})
    con.send(obj)
    res = con.recv(BUFFSIZE)
    res = json.loads(res)
    if res['status'] != 'ready':
        return None
    
    # serialize and send model to registry
    m = pickle.dumps(model)
    con.send(m)
    res = con.recv(BUFFSIZE)
    res = json.loads(res)

    # return model id
    if res['status'] == 'completed':
        return res['model_id']
    return None

def get_model(con, model_id):
    ''' Gets the model from the registry. '''
    model = 'binary'
    return model
