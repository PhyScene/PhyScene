from typing import Dict, List
import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry
from models.optimizer.optimizer import Optimizer

OPTIMIZER = Registry('Optimizer')


def create_optimizer(cfg: DictConfig, *args: List, **kwargs: Dict) -> Optimizer:
    """ Create a optimizer for constrained sampling

    Args:
        cfg: configuration object
    
    Return:
        A optimizer used for guided sampling
    """
    if cfg is None:
        return None
    
    return OPTIMIZER.get(cfg.name)(cfg, *args, **kwargs)


