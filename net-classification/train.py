import argparse
import os
import warnings
from pathlib import Path

import albumentations as A
import torch
from fvcore.common.file_io import PathManager
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from timm.models import load_checkpoint
from timm.utils import *

from src.dataset import LoadDatasets
from src.defaults import get_cfg
from src.lightning import ModelClass, VerboseCallback
from src.models import build_model
from src.utils import colorstr

def merge_config(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    return cfg


def run(args):
    fold = args.fold

    cfg = merge_config(args)
    out_dir = Path(cfg.OUTPUT_DIR)
    run_dir = out_dir / cfg.RUN_NAME
    run_dir = run_dir / f"fold_{fold}"
    PathManager.mkdirs(path=run_dir)

    seed_everything(cfg.SEED, True)

    model = build_model(cfg)

    print(colorstr(f"** Training on FOLD : {fold}"))

    if cfg.CHECKPOINT:
        print(f"Attempting to load checkpoint from : {cfg.CHECKPOINT}")
        assert PathManager.isfile(path=cfg.CHECKPOINT)
        load_checkpoint(model, cfg.CHECKPOINT)
        print(f"Checkpoint loaded ....")

    lit_model = ModelClass(model, cfg=cfg, run_dir=run_dir)

    train_augs = None

    if cfg.INPUT.TRAIN_AUGS != " ":
        assert PathManager.isfile(path=cfg.INPUT.TRAIN_AUGS)
        train_augs = A.load(cfg.INPUT.TRAIN_AUGS, data_format="json")

    datamodule = LoadDatasets(cfg, fold, train_augs, None)

    cp = os.path.join(run_dir, "config.yaml")

    with PathManager.open(cp, "w") as f:
        f.write(cfg.dump())

    trainer = Trainer(
        logger=[CSVLogger(run_dir),],
        callbacks=[LearningRateMonitor("step"), VerboseCallback(run_dir),],
        deterministic=True,
        checkpoint_callback=False,
        **cfg.TRAINER)
    
    trainer.fit(lit_model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_default_logging()

    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--fold", default=0, type=int, required=True)
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    print(f'{colorstr("Command Line Args: ")}{args}')
    
    warnings.filterwarnings("ignore")
    
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.deterministic = True

    run(args)
