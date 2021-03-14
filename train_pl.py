import os
import argparse
import multiprocessing as mp

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torch_pointnet.datasets
from torch_pointnet.losses import PointNetLoss
from torch_pointnet.models import PointNetClassifier
from torch_pointnet.transforms import train_transforms, test_transforms


# Make reproducible
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Module(pl.LightningModule):

    def __init__(model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.model.parameters(),
            lr=cfg.train.lr
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, input_matrix, feature_matrix = self(x)
        loss = self.loss_fn(y_pred, y, input_matrix, feature_matrix)
        acc = pl.metrics.accuracy(y_pred, y)
        self.log("loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred, input_matrix, feature_matrix = self(x)
        loss = self.loss_fn(y_pred, y, input_matrix, feature_matrix)
        acc = pl.metrics.accuracy(y_pred, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred, input_matrix, feature_matrix = self(x)
        loss = self.loss_fn(y_pred, y, input_matrix, feature_matrix)
        acc = pl.metrics.accuracy(y_pred, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss


def main(cfg):

    # Load model
    model = PointNetClassifier(
        num_points=cfg.data.num_points,
        num_classes=cfg.data.num_classes
    )
    model = model.to(cfg.device)

    # Setup optimizer and loss func
    loss_fn = PointNetLoss(weight_decay=cfg.train.weight_decay)

    # Load datasets
    dataset = getattr(torch_pointnet.datasets, cfg.data.dataset)

    train_dataset = dataset(
        root=cfg.data.root,
        split="train",
        transforms=train_transforms(
            num_points=cfg.data.num_points,
            mu=cfg.data.transforms.jitter_mean,
            sigma=cfg.data.transforms.jitter_std
        )
    )
    test_dataset = dataset(
        root=cfg.data.root,
        split="test",
        transforms=test_transforms(
            num_points=cfg.data.num_points
        )
    )

    # Setup dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=mp.cpu_count(),
        prefetch_factor=cfg.train.num_prefetch
    )

    module = Module(model, loss_fn)
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        gpus=1,
        precision=16
    )
    trainer.fit(
        module,
        train_dataloader,
        test_dataloader
    )
    trainer.test(module, test_dataloader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="Path to config.yaml file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.cfg)
    main(cfg)