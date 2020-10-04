import os
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from sklearn.model_selection import StratifiedKFold
from torch import nn, optim
from torchvision.models import resnet18

try:
    from dataset import ChineseMNISTDataModule, ChineseMNISTDataset
except:
    pass


class ChineseMNISTResnetModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = 15
        resnet = resnet18(pretrained=True, progress=True)
        resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            dilation=resnet.conv1.dilation,
            bias=resnet.conv1.bias,
        )
        resnet.fc = nn.Linear(512, self.num_classes)
        self.resnet = resnet
        self.accuracy = Accuracy(num_classes=self.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image):
        image = image.permute(0, 3, 1, 2).contiguous().float()
        return self.resnet(image)

    def training_step(self, batch, batch_idx: int):
        image, y = batch
        yhat = self(image)
        loss = self.criterion(yhat, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx: int):
        image, y = batch
        yhat = self(image)
        loss = self.criterion(yhat, y)
        acc = self.accuracy(yhat, y)
        return {"val_loss": loss, "val_acc": acc, "progress_bar": {"val_acc": acc}}

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        return {"test_acc": metrics["val_acc"], "test_loss": metrics["val_loss"]}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def training(k_folds: int = 5):
    is_kaggle = os.path.isdir("/kaggle")
    data_root = Path("/kaggle/input/chinese-mnist" if is_kaggle else "archive")
    all_df = pd.read_csv(data_root / "chinese_mnist.csv")

    skf = StratifiedKFold(n_splits=k_folds)

    trainer = pl.Trainer(gpus=1, max_epochs=5, precision=16)

    for train_indices, val_indices in skf.split(all_df, all_df.code):
        data_module = ChineseMNISTDataModule(
            data_root=data_root,
            all_df=all_df,
            train_indices=train_indices,
            val_indices=val_indices,
        )
        model = ChineseMNISTResnetModel()
        trainer.fit(model, data_module)


if __name__ == "__main__":
    training()
