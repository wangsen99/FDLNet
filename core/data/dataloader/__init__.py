"""
This module provides data loaders and transformers for popular vision datasets.
"""
from .nightcity import NightSegmentation
from .cityscapes import CitySegmentation


datasets = {
    'night': NightSegmentation,
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
