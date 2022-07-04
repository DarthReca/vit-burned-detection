# Copyright 2022 Daniele Rege Cambrin
import os.path
import warnings
from itertools import chain
from typing import Dict, Any, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler

from utils import (
    config_to_object,
    SatelliteDataset,
    AggregationDataset,
)

VALIDATION_DICT = {
    "purple": "coral",
    "coral": "cyan",
    "pink": "coral",
    "grey": "coral",
    "cyan": "coral",
    "lime": "coral",
    "magenta": "coral",
}


class LitDataModule(pl.LightningDataModule):
    def __init__(self, **hparams: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()

        self.train_transforms = [
            config_to_object("torchvision.transforms", k, v)
            for k, v in self.hparams["train_transform"].items()
        ]

        self.test_transforms = [
            config_to_object("torchvision.transforms", k, v)
            for k, v in self.hparams["test_transform"].items()
        ]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.generator = torch.Generator().manual_seed(self.hparams["seed"])

        self.batch_size = self.hparams["batch_size"]
        self.test_fold = self.hparams["key"]

    def prepare_data(self) -> None:
        if not os.path.isdir("data/RESCUE"):
            warnings.warn("dataset non in data/RESCUE")

    def setup(self, stage: Optional[str] = None) -> None:
        validation_fold_name = VALIDATION_DICT[self.test_fold]
        print(
            f"Test set is "
            f"{self.hparams['key']}, validation set is {validation_fold_name}. All the rest is training set."
        )
        dataset_class = (
            AggregationDataset if self.hparams["aggregate"] else SatelliteDataset
        )
        if stage in ("fit", None):
            train_set = list(
                chain(
                    *[
                        self.hparams["groups"][grp]
                        for grp in self.hparams.groups
                        if grp != validation_fold_name and grp != self.test_fold
                    ]
                )
            )
            self.train_dataset = dataset_class(
                folder_list=train_set,
                transform=self.train_transforms,
                **self.hparams,
            )
        if stage in ("fit", "validate", None):
            validation_set = self.hparams["groups"][validation_fold_name]
            self.val_dataset = dataset_class(
                folder_list=validation_set,
                transform=self.test_transforms,
                **self.hparams,
            )
        if stage in ("test", None):
            test_set = self.hparams["groups"][self.test_fold]
            self.test_dataset = dataset_class(
                folder_list=test_set,
                transform=self.test_transforms,
                **self.hparams,
            )

    def train_dataloader(self):
        print(f"Training set is {len(self.train_dataset)} lenght")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            sampler=RandomSampler(self.train_dataset, generator=self.generator),
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        print(f"Test set is {len(self.test_dataset)} lenght")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )

    def val_dataloader(self):
        print(f"Validation set is {len(self.val_dataset)} lenght")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=False,
        )
