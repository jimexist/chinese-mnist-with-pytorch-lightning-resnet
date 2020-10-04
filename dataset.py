import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset


class ChineseMNISTDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df

    def __len__(self):
        return len(self.df)


class ChineseMNISTDataModule(pl.DataModule):
    def __init__(self, data_root: str) -> None:
        super().__init__()

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass


if __name__ == "__main__":
    is_kaggle = os.path.isdir("/kaggle")
    data_root = Path("/kaggle/input/chinese-mnist" if is_kaggle else "archive")
    assert os.path.isdir(data_root), f"{data_root} is not a dir"

    df = pd.read_csv(data_root / "chinese_mnist.csv")
    dataset = ChineseMNISTDataset(df)
