import importlib
import torch
from src.pose_lib.models import rotatespade_model as modellib


def find_model_using_name(model_name):

    model = modellib.RotateSPADEModel

    return model


def get_option_setter(model_name):
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))

    return instance
