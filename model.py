import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset

from dataset import ChineseMNISTDatamodule, ChineseMNISTDataset


class ChineseMNISTDataset(Dataset):
    def __init__(self, df):
        super().__init__()
