import json
import logging
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import *

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_toolbelt.losses import BinaryFocalLoss, JointLoss
from timm.loss.asymmetric_loss import AsymmetricLossSingleLabel
from timm.utils import CheckpointSaver
from torch.optim import optimizer
from torchmetrics.classification.auroc import AUROC
from torchmetrics.classification.average_precision import AveragePrecision
from yacs.config import CfgNode

from .solver import LOSSES_DICT, build_lr_scheduler, build_optimizer
from .utils import colorstr


class ModelClass(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg: CfgNode, run_dir: str):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = model
        self.auc = AUROC(num_classes=cfg.INPUT.NUM_CLASSES)
        self.prec = AveragePrecision(num_classes=cfg.INPUT.NUM_CLASSES)
        self.cls_loss = LOSSES_DICT[cfg.TRAINING.LOSS.CLASS_LOSS]

        if hasattr(self.model, "mask_head") and cfg.TRAINING.AUX_LOSS:
            if cfg.TRAINING.LOSS.COMBO_LOSS.USE:
                loss_1 = LOSSES_DICT[cfg.TRAINING.LOSS.COMBO_LOSS.L1]
                loss_2 = LOSSES_DICT[cfg.TRAINING.LOSS.COMBO_LOSS.L2]
                weights = cfg.TRAINING.LOSS.COMBO_LOSS.WEIGHTS
                self.aux_loss = JointLoss(loss_1, loss_2, weights[0], weights[1])
            else:
                self.aux_loss = LOSSES_DICT[cfg.TRAINING.LOSS.MASK_LOSS]
        else:
            self.aux_loss = None
        self._cfg = cfg

        if self._cfg.TRAINING.EMA:
            # mprint(colorstr("Applying Model EMA"))
            self.ema = deepcopy(self.model).eval()
            for p in self.ema.parameters():
                p.requires_grad_(False)
            self.updates = 0
            # decay exponential ramp (to help early epochs)
            self.decay = lambda x: 0.9999 * (1 - math.exp(-x / 2000))

        self.run_dir = run_dir

    def forward(self, input):
        return self.model(input)

    @property
    def num_training_steps(self) -> int:
        dataset_size = len(self.trainer.datamodule.train_dataloader())
        num_devices = max(1, self.trainer.num_gpus)
        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (dataset_size // effective_batch_size) * self.trainer.max_epochs
        return max_estimated_steps

    def configure_optimizers(self):
        self.opt = optimizer = build_optimizer(self._cfg, self, self.num_training_steps)
        lr_scheduler = build_lr_scheduler(self._cfg, optimizer, self.num_training_steps)
        if lr_scheduler is None:
            return [optimizer]
        else:
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        if self.trainer.current_epoch == 0:
            self.saver = CheckpointSaver(
                self.model,
                self.opt,
                checkpoint_dir=self.run_dir,
                decreasing=False,
                max_history=1,
                unwrap_fn=lambda model: model,
            )

    def unpack_batch(self, batch: Dict):
        input = batch["image"]
        target = batch["target"]
        mask = batch.get("mask", None)
        image_id = batch["image_id"]
        return input, target, mask, image_id

    def training_step(self, batch, batch_idx):
        total_loss = 0 # Initialize total loss

        # 1. Unpack components of a single training batch
        input, target, mask, _ = self.unpack_batch(batch)
        output = self(input)

        # 2. Compute logits
        logits = output["output"]

        # 3. Compute normal classification loss
        cls_loss = self.cls_loss(logits, target)

        # 4. Compute Distillation loss if noisy labels are given
        if batch.get("ns_label", None) is not None:
            distill_loss = F.binary_cross_entropy_with_logits(logits, batch["ns_label"])
            self.log("train/distill_loss", distill_loss)
            cls_loss = cls_loss * (1 - 0.5) + distill_loss * 0.5

        total_loss += cls_loss
        self.log("train/cls_loss", cls_loss)

        # 5. Compute auxiliary on segmentation masks (if applicable)
        if self.aux_loss is not None:
            aux_loss = self.aux_loss(output["mask"], mask)
            aux_loss *= self._cfg.TRAINING.AUX_LOSS_WEIGHT
            total_loss += aux_loss
            self.log("train/aux_loss", aux_loss)

        self.log("train/total_loss", total_loss)
        return total_loss

    def update_ema(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def on_train_batch_end(self, *args, **kwargs):
        if hasattr(self, "ema"):
            self.update_ema(self.model)

    def shared_eval_step(self, batch, prefix):
        input, target, _, _ = self.unpack_batch(batch)

        if hasattr(self, "ema"): output = self.ema(input)["output"]
        else                   : output = self(input)["output"]
        loss = self.cls_loss(output, target)

        self.log(f"{prefix}/loss", loss)
        return {"scores": F.softmax(output), "labels": target}

    def shared_eval_epoch_end(self, outputs, prefix):
        scores_all     = torch.cat([output["scores"] for output in outputs]).float()
        labels_one_hot = torch.cat([output["labels"] for output in outputs])
        labels_all     = torch.argmax(labels_one_hot, dim=1).long()

        roc_auc = self.auc(scores_all, labels_one_hot.long())

        av_prec = self.prec(scores_all, labels_all)
        av_prec = torch.mean(torch.stack(av_prec))

        self.log(f"{prefix}/map", av_prec)
        self.log(f"{prefix}/roc_auc", roc_auc)
        self.log(f"{prefix}/map*0.6", av_prec * (2 / 3))

        if prefix == "valid": self.custom_save(av_prec * (2 / 3))

    @rank_zero_only
    def custom_save(self, metric):
        best_metric, best_epoch = self.saver.save_checkpoint(epoch=self.trainer.current_epoch, metric=metric)
        self.print(f"Best Epoch ({best_metric}) Best Metric ({best_epoch})")

    def validation_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, prefix="valid")

    def validation_epoch_end(self, outputs):
        self.shared_eval_epoch_end(outputs, prefix="valid")

    def test_step(self, batch, batch_idx):
        return self.shared_eval_step(batch, prefix="test")

    def test_epoch_end(self, outputs):
        self.shared_eval_epoch_end(outputs, prefix="test")


class VerboseCallback(Callback):
    def __init__(self, output_dir: str) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: LightningModule):
        logged_metrics = trainer.logged_metrics
        current_epoch = trainer.current_epoch

        if not trainer.sanity_checking:
            summary = (
                "{header}  Epoch ({current_epoch:>2d}/{trainer.max_epochs:>2d})  "
                "AUROC: {auc:>6.4f} | "
                "AP: {map:>6.4f} | "
                "(AP*0.66): {map_06:>6.4f} | "
                "valid/loss: {valid_loss:>6.4f} | "
                "train/loss: {train_loss:>6.4f} ".format(
                    header=colorstr("green", "bold", f"* {time.ctime()}"),
                    trainer=trainer,
                    current_epoch=current_epoch,
                    auc=logged_metrics["valid/roc_auc"],
                    train_loss=logged_metrics["train/total_loss"],
                    valid_loss=logged_metrics["valid/loss"],
                    map=logged_metrics["valid/map"],
                    map_06=logged_metrics["valid/map*0.6"],
                )
            )
            pl_module.print(summary)
            log_stats = trainer.logged_metrics
            log_stats = {k: v.data.cpu().numpy().item() for k, v in log_stats.items()}
            log_stats["epoch"] = current_epoch
            log_stats["global_step"] = pl_module.global_step
            self.write_json(log_stats)

    @rank_zero_only
    def write_json(self, log_stats):
        with (self.output_dir / "summary.json").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
