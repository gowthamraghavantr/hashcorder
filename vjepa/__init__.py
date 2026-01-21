"""
V-JEPA: Vision Joint-Embedding Predictive Architecture

A minimal, research-grade implementation of Meta's V-JEPA for self-supervised
vision representation learning.
"""

from . import models
from . import data
from . import config
from . import utils

from .models import VJEPA, VJEPALoss, build_encoder, build_predictor
from .config import ExperimentConfig, get_cnn_config, get_vit_config, get_debug_config
from .data import create_simple_dataloader

__version__ = "0.1.0"

__all__ = [
    'models',
    'data',
    'config',
    'utils',
    'VJEPA',
    'VJEPALoss',
    'build_encoder',
    'build_predictor',
    'ExperimentConfig',
    'get_cnn_config',
    'get_vit_config',
    'get_debug_config',
    'create_simple_dataloader',
]
