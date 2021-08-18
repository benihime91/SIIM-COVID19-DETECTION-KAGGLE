import math
import os
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from yacs.config import CfgNode

TRAIN_TRANSFORM = albu.Compose([albu.NoOp()])

VALID_TRANSFORM = albu.Compose([albu.NoOp()])


def get_class_names(nc: int):
    if nc == 4:
        return ["atypical", "indeterminate", "negative", "typical"]
    elif nc == 2:
        return ["opacity", "negative"]


def get_tag2id(nc: int):
    if nc == 4:
        return {0: "atypical", 1: "indeterminate", 2: "negative", 3: "typical"}
    elif nc == 2:
        return {0: "opacity", 1: "negative"}


class LoadImagesLabels(Dataset):
    def __init__(
        self, cfg: CfgNode, df, transforms: albu.Compose, mode="train", noisy_df=None
    ):
        # Setup path, datasets, noisy labels
        self.data       = df.reset_index(drop=True)
        self.image_path = Path(cfg.PATH_IMAGES)
        self.mask_path  = Path(cfg.PATH_MASK)

        if noisy_df is not None: noisy_df = noisy_df.reset_index(drop=True) 
        self.noisy_df   = noisy_df

        self.cfg = cfg

        # check paths
        assert os.path.exists(self.image_path)
        assert os.path.exists(self.mask_path)


        self.num_classes = cfg.INPUT.NUM_CLASSES
        self.class_names = get_class_names(self.num_classes)
        self.tag2id = get_tag2id(self.num_classes)

        self.return_masks = (
            cfg.TRAINING.AUX_LOSS
            and mode == "train"
            and cfg.MODEL.AUX_MODEL.VERSION == "v1"
        )

        self.mode = mode

        self.transforms: albu.Compose = transforms

        # Common preprocessing transformations for all
        self.preprocessing = albu.Compose([
            albu.Resize(self.cfg.INPUT.INPUT_SIZE, self.cfg.INPUT.INPUT_SIZE),
            albu.ToFloat(p=1.0, max_value=255.0),
            ToTensorV2(p=1.0),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        image_id = row.id
        try:
            image_path = self.image_path / f"{image_id}.png"
            assert os.path.isfile(image_path)
        except:
            image_path = self.image_path / f"{image_id}.jpg"
            assert os.path.isfile(image_path)
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        data_dict = {}
        data_dict["image_id"] = image_id

        if self.return_masks:
            mask_path = os.path.join(self.mask_path, f"{image_id}.png")
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.zeros((h, w), dtype=np.uint8)

        aug_data = self.transforms(image=image, mask=mask)
        image = aug_data["image"]
        aux_mask = aug_data["mask"]

        image = self.preprocessing(image=image)["image"]
        data_dict["image"] = image

        aux_mask = albu.resize(
            np.array(aux_mask, np.float32),
            math.ceil(self.cfg.INPUT.INPUT_SIZE / 16),
            math.ceil(self.cfg.INPUT.INPUT_SIZE / 16),
        )
        aux_mask = aux_mask / 255.0
        aux_mask = np.expand_dims(aux_mask, -1).transpose(2, 0, 1)
        data_dict["mask"] = aux_mask

        target = [row[c] for c in self.class_names]
        data_dict["target"] = np.array(target, np.float32)

        if self.noisy_df is not None:
            row = self.noisy_df.iloc[index]
            nst = [row[c] for c in self.class_names]
            data_dict["ns_label"] = np.array(nst, np.float32)
        return data_dict


class LoadDatasets(LightningDataModule):
    def __init__(self, cfg: CfgNode, fold: int, train_tfms=None, valid_tfms=None):
        super().__init__()
        self.cfg = cfg
        self.fold = fold

        # 1. Load augmentations for training and validation datasets
        self.train_tfms = TRAIN_TRANSFORM if train_tfms is None else train_tfms
        self.valid_tfms = VALID_TRANSFORM if valid_tfms is None else valid_tfms

        # 2. Load the base pandas dataframes
        self.data_frame = pd.read_csv(self.cfg.PATH_CSV) # original data
        self.noisy_frame = pd.read_csv(cfg.NOISY_CSV) if cfg.NOISY_CSV else None # noisy data with soft labels

        # 3. Grab the data for the current fold
        self.trn_data = self.data_frame.query(f"kfold != {self.fold}").reset_index(drop=True, inplace=False)
        self.val_data = self.data_frame.query(f"kfold == {self.fold}").reset_index(drop=True, inplace=False)
        if self.noisy_frame is not None:
            self.noisy_data = self.noisy_frame.query(f"kfold != {self.fold}").reset_index(drop=True, inplace=False)
        else:
            self.noisy_data = None

        # 4. Parse config
        self.batch_size = self.cfg.DATALOADER.BATCH_SIZE
        self.workers = self.cfg.DATALOADER.NUM_WORKERS
        self.tag2id = get_tag2id(cfg.INPUT.NUM_CLASSES)

        # 5. Add Pseudo labels and training data if given
        if self.cfg.PATH_PSEUDO is not None:
            self.pseudo_data = pd.read_csv(self.cfg.PATH_PSEUDO)
            self.trn_data = pd.concat([self.trn_data, self.pseudo_data])
            print(f'Loaded pseudo labels from {self.cfg.PATH_PSEUDO}')
            self.trn_data = self.trn_data.sample(frac=1).reset_index(inplace=False, drop=True)

    def setup(self, stage=None):
        '''Setup Training and Validation datasets'''
        self.trn_dataset = LoadImagesLabels(
            self.cfg, self.trn_data,
            self.train_tfms, mode="train", noisy_df=self.noisy_data)

        self.val_dataset = LoadImagesLabels(
            self.cfg, self.val_data, self.valid_tfms, mode="valid", noisy_df=None)

    def train_dataloader(self):
        return DataLoader(self.trn_dataset, self.batch_size, shuffle=True, num_workers=self.workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.workers)

    def predict_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=self.workers)