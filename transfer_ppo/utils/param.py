### Store the parameters
import os
import torch

class Param:
    dtype = None
    device = None
    # get absolute dir
    root_dir = os.getcwd()
    model_dir = 'learned_models/'
    plot_dir  = 'plot/'
    log_dir = 'results/'
    data_dir = 'data/'
    logger_file = None
    def __init__(self, dtype=None, device=None):
        if dtype is not None:
            Param.dtype = dtype
        if device is not None:
            Param.device = device
    def get():
        return (Param.dtype, Param.device)