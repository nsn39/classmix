import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.voc_dataset import VOCDataSet
from data.natural_loader import NaturalDatasetLoader

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "pascal_voc": VOCDataSet,
        "natural": NaturalDatasetLoader
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '/content/drive/MyDrive/CityScapes/'
    if name == 'pascal_voc':
        return '/content/classmix/data/VOC2012/'
    if name == 'natural':
        return '/content/dirve/MyDrive/natural/natural-images'