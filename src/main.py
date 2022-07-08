from .ConvNeXt_stages import ConvNeXt_stages
from .ConvNeXt_stem import ConvNeXt_stem
from .FULLCONV import FULLCONV
from .LayerScaler import LayerScaler
from .Reg_DataAugmentation.py import Reg_DataAugmentation
from .ResNet_block import ResNet_block

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor


def main():
	"""
	File to complete after correcting the training
	error
	"""