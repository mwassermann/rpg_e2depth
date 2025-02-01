import torch
from model.model import *


def load_model(path_to_model):
    print('Loading model {}...'.format(path_to_model))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    raw_model = torch.load(path_to_model, map_location=device)
    arch = raw_model['arch']

    try:
        model_type = raw_model['model']
    except KeyError:
        model_type = raw_model['config']['model']

    print ("Model Type", model_type)
    # instantiate model
    model = eval(arch)(model_type)

    # load model weights
    model.load_state_dict(raw_model['state_dict'])

    return model


def get_device(use_gpu):
    device = torch.device('cuda:0') if use_gpu and torch.cuda.is_available() else torch.device('mps') if use_gpu and torch.backends.mps.is_available() else torch.device('cpu')

    print('Device:', device)

    return device
