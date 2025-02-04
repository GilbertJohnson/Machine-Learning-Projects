import random
import numpy as np
import torch
from typing import Tuple
from torchvision import transforms as T
import torch.nn as nn

from dataclasses import dataclass
class BuildNN:
    def __init__(this,seed_value:int=42,config:dict=None):
        this.set_seeds(SEED_VALUE=seed_value)
        this.config = config
    
    @staticmethod
    def set_seeds(SEED_VALUE:int=42)->None:
        random.seed(SEED_VALUE)
        np.random.seed(SEED_VALUE)
        torch.manual_seed(SEED_VALUE)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED_VALUE)
            torch.cuda.manual_seed_all(SEED_VALUE)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True