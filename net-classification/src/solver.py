from typing import *

import torch
from fvcore.common.param_scheduler import *
from ranger21 import Ranger21
from ranger.ranger2020 import Ranger
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from yacs.config import CfgNode
from functools import partial


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    p.grad.norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm


# ------------------------------ optimizer factory -------------------------------- #
def build_optimizer(
    cfg: CfgNode, model: torch.nn.Module, steps
) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = filter(lambda p: p.requires_grad, model.parameters())
    name = cfg.TRAINING.OPTIMIZER.NAME
    if name == "Ranger":
        opt_func = Ranger
    elif name == "Ranger21":
        opt_func = partial(
            Ranger21, num_batches_per_epoch=steps, num_epochs=cfg.TRAINER.max_epochs
        )
    elif name == "AdamW":
        opt_func = AdamW
    else:
        raise ValueError("Unkonwn optimizer.")

    kwargs = dict(cfg.TRAINING.OPTIMIZER.INIT_ARGS) if name == "Ranger21" else {}

    if cfg.TRAINING.OPTIMIZER.USE_SAM:
        base_opt = partial(
            opt_func,
            lr=cfg.TRAINING.OPTIMIZER.LR,
            weight_decay=cfg.TRAINING.OPTIMIZER.WEIGHT_DECAY,
            betas=cfg.TRAINING.OPTIMIZER.BETAS,
            eps=cfg.TRAINING.OPTIMIZER.EPS,
            **kwargs,
        )
        optimizer = SAM(params, base_opt)
        print(
            f"SAM optimizer loaded with base : {optimizer.base_optimizer.__class__.__name__}"
        )

    else:
        optimizer = opt_func(
            params,
            lr=cfg.TRAINING.OPTIMIZER.LR,
            weight_decay=cfg.TRAINING.OPTIMIZER.WEIGHT_DECAY,
            betas=cfg.TRAINING.OPTIMIZER.BETAS,
            eps=cfg.TRAINING.OPTIMIZER.EPS,
            **kwargs,
        )
    return optimizer


# ------------------------------ Learning Rate Schedulers ---------------------------- #
class LRMultiplier(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, max_iter, last_iter=-1):
        if not isinstance(multiplier, ParamScheduler):
            raise ValueError(
                "_LRMultiplier(multiplier=) must be an instance of fvcore "
                f"ParamScheduler. Got {multiplier} instead."
            )
        self._multiplier = multiplier
        self._max_iter = max_iter
        super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]


class WarmupParamScheduler(CompositeParamScheduler):
    def __init__(self, scheduler, warmup_factor, warmup_length, warmup_method="linear"):
        end_value = scheduler(warmup_length)
        start_value = warmup_factor * scheduler(0.0)
        if warmup_method == "constant":
            warmup = ConstantParamScheduler(start_value)
        elif warmup_method == "linear":
            warmup = LinearParamScheduler(start_value, end_value)
        else:
            raise ValueError("Unknown warmup method: {}".format(warmup_method))
        super().__init__(
            [warmup, scheduler],
            interval_scaling=["rescaled", "fixed"],
            lengths=[warmup_length, 1 - warmup_length],
        )


# --------------------------- SCHEDULER FACTORY ------------------------------------ #
WARMUP_FACTOR = 1e-05


def cosine_scheduler(optimizer, pct_start, max_iters, warmup_pct):
    """
    Cosine scheduler with warmup , hold and warmdown pcts
    """
    if warmup_pct > 0.0:
        mul = ConstantParamScheduler(1.0)
        warmup_scheduler = WarmupParamScheduler(
            mul, WARMUP_FACTOR, warmup_pct, "linear",
        )
        schedulers = [warmup_scheduler, CosineParamScheduler(1, 0)]
        lengths = [pct_start, 1 - pct_start]
    else:
        schedulers = [LinearParamScheduler(1, 1), CosineParamScheduler(1, 1e-05)]
        lengths = [pct_start, 1 - pct_start]

    sched = CompositeParamScheduler(schedulers, lengths, ["rescaled"] * len(lengths))
    sched = LRMultiplier(optimizer, sched, max_iter=max_iters)
    return sched


def linear_scheduler(optimizer, max_iters, warmup_steps):
    """Linearly increase `lr` for `warmup_steps` before Linear annealing"""
    warmup_length = warmup_steps / max_iters
    sched = LinearParamScheduler(1, 1e-05)
    sched = WarmupParamScheduler(sched, WARMUP_FACTOR, warmup_length, "linear")
    return LRMultiplier(optimizer, multiplier=sched, max_iter=max_iters)


def build_lr_scheduler(cfg: CfgNode, optimizer, training_steps):
    """
    Build a LR scheduler from config.
    """
    name = cfg.TRAINING.SCHEDULER.NAME
    total_steps = training_steps + 50

    if name is None:
        return None

    elif name == "OneCycle":
        max_lr = cfg.TRAINING.SCHEDULER.MAX_LR
        scheduler_kwargs = {}
        scheduler_kwargs["max_lr"] = max_lr
        scheduler_kwargs["total_steps"] = total_steps
        return OneCycleLR(optimizer, **scheduler_kwargs)

    elif name == "Cosine":
        pct_start = cfg.TRAINING.SCHEDULER.WARMDOWN_PCT
        warmup_stp = cfg.TRAINING.SCHEDULER.WARMUP_EPOCHS
        warmup_pct = warmup_stp / total_steps
        return cosine_scheduler(optimizer, pct_start, total_steps, warmup_pct)

    elif name == "Linear":
        warmup_steps = cfg.TRAINING.SCHEDULER.WARMUP_EPOCHS
        return linear_scheduler(optimizer, total_steps, warmup_steps)

    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))


# ---------------------------------------------------
# Scikit-Learn metric
import numpy as np
from sklearn.metrics import average_precision_score


def sklearn_metric(probability, truth):
    # @ https://www.kaggle.com/c/siim-covid19-detection/discussion/246783
    _, num_label = probability.shape
    score = []
    for i in range(num_label):
        s = average_precision_score(truth == i, probability[:, i])
        score.append(s)
    score = np.array(score)
    score = np.mean(score)  # * 2 / 3
    return score


# -----------------------------------
# Loss function Libarary
from pytorch_toolbelt.losses import (
    BinaryLovaszLoss,
    DiceLoss,
    BinaryFocalLoss,
)
from torch.nn import BCEWithLogitsLoss
from timm.loss import SoftTargetCrossEntropy

LOSSES_DICT = dict(
    cross_entropy=SoftTargetCrossEntropy(),
    focal_loss=BinaryFocalLoss(alpha=0.25, gamma=2),
    bce_loss=BCEWithLogitsLoss(),
    lovasz_loss=BinaryLovaszLoss(),
    dice_loss=DiceLoss(mode="binary", classes=None, from_logits=True),
)

