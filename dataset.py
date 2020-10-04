import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn
from torch.utils.data import DataLoader, Dataset


class ChineseMNISTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Path,
        transform: A.BasicTransform = None,
    ) -> None:
        super().__init__()
        self.df = df
        self.image_root = image_root
        self.transform = transform

    def __getitem__(self, idx: int):
        row = self.df.loc[idx, :]
        suite_id, code, sample_id = row.suite_id, row.code, row.sample_id
        assert 1 <= code <= 15, f"invalid code {code}"
        filename = self.image_root / f"input_{suite_id}_{sample_id}_{code}.jpg"
        assert os.path.isfile(filename), f"{filename} is not a file"
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, np.newaxis]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, code - 1

    def __len__(self):
        return len(self.df)


class ChineseMNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: Path,
        all_df: pd.DataFrame,
        train_indices: pd.Index,
        val_indices: pd.Index,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.df = all_df
        self.image_root = self.data_root / "data" / "data"
        self.train_df = self.df.loc[train_indices, :].copy().reset_index()
        self.train_transform = A.Compose(
            [
                ToTensorV2(),
            ]
        )
        self.val_df = self.df.loc[val_indices, :].copy().reset_index()
        self.val_transform = A.Compose(
            [
                ToTensorV2(),
            ]
        )

    def train_dataloader(self):
        ds = ChineseMNISTDataset(self.train_df, self.image_root, self.train_transform)
        return DataLoader(
            ds,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        ds = ChineseMNISTDataset(self.val_df, self.image_root, self.val_transform)
        return DataLoader(
            ds,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )


if __name__ == "__main__":
    is_kaggle = os.path.isdir("/kaggle")
    data_root = Path("/kaggle/input/chinese-mnist" if is_kaggle else "archive")
    assert os.path.isdir(data_root), f"{data_root} is not a dir"
    df = pd.read_csv(data_root / "chinese_mnist.csv")

    data_module = ChineseMNISTDataModule(data_root, df, df.index[:20], df.index[20:30])
