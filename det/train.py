#!/usr/bin/env python
import ast
import datetime
import json
import math
import os
import random
import sys
import time
import warnings
from contextlib import suppress
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from albumentations.pytorch import ToTensorV2
from effdet import create_model, unwrap_bench
from timm.models.helpers import load_checkpoint
from timm.utils import (
    CheckpointSaver,
    ModelEmaV2,
    NativeScaler,
    get_outdir,
    setup_default_logging,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

import coco_eval
import coco_utils
import utils

bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup(im, labels, im2, labels2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def get_augmentations(size):
    """Training/ Validation Augmentations"""
    train_presets = albu.Compose(
        [
            albu.Resize(height=size, width=size, p=1.0),
            albu.HorizontalFlip(p=0.5),
            albu.ShiftScaleRotate(
                rotate_limit=25, p=0.5, border_mode=cv2.BORDER_REFLECT
            ),
            albu.RandomBrightnessContrast(0.20, 0.20, p=0.5),
            ToTensorV2(p=1.0),
        ],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )

    valid_presets = albu.Compose(
        [albu.Resize(height=size, width=size, p=1.0), ToTensorV2(p=1.0),],
        bbox_params=albu.BboxParams(format="pascal_voc", label_fields=["labels"]),
    )
    return train_presets, valid_presets


def set_bn_eval(m: nn.Module, use_eval=True) -> None:
    "Set bn layers in eval mode for all recursive children of `m`."
    for l in m.children():
        if isinstance(l, bn_types) and not next(l.parameters()).requires_grad:
            if use_eval:
                l.eval()
            else:
                l.train()
        set_bn_eval(l)


class LoadImagesAndLabels(Dataset):
    def __init__(
        self,
        root,
        frame,
        transforms,
        imsize,
        size: int,
        mosaic=0.0,
        mixup=0.0,
        test=False,
    ):
        self.root = Path(root)
        self.data = frame
        self.imsize = imsize
        self.mosaic = mosaic
        self.test = test
        self.transforms = transforms
        self.mosaic_border = [-imsize // 2, -imsize // 2]
        self.size = size
        self.mixup = mixup

        if mosaic and not test:
            print("-- mosaic augmentation = {}".format(self.mosaic))
        if mixup and not test:
            print("-- mixup augmentation = {}".format(self.mixup))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        image_id = self.data["image_id"][index]
        mosaic = self.mosaic and random.random() < self.mosaic

        if self.test:
            image, boxes, _ = self.load_image_and_boxes(index)
        else:
            if mosaic:
                # Load mosaic
                image, boxes = self.load_mosaic_image_and_boxes(index)
                # MixUp augmentation
                if self.mixup and random.random() < self.mixup:
                    idx = random.randint(0, len(self) - 1)
                    image, boxes = mixup(
                        image, boxes, *self.load_mosaic_image_and_boxes(idx)
                    )
            else:
                image, boxes, _ = self.load_image_and_boxes(index)

        labels = np.ones((boxes.shape[0],), dtype=np.int64)

        o_image, o_boxes = image.copy(), boxes.copy()
        no_boxes = False

        for _ in range(50):
            sample = self.transforms(image=image, bboxes=boxes, labels=labels.tolist())
            if len(sample["bboxes"]) > 0:
                image = sample["image"]
                boxes = sample["bboxes"]
                boxes = np.array(boxes)
                no_boxes = False
                break
            else:
                no_boxes = True

        if no_boxes:
            image = albu.resize(o_image, self.size, self.size)
            image = torch.from_numpy(image.transpose(2, 0, 1))
            boxes = np.array(o_boxes)
            warnings.warn(f"Cannot augment Image : {image_id}")

        # xyxy -> yxyx
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # yxyx

        annotation = {}
        annotation["bbox"] = torch.as_tensor(boxes, dtype=torch.float32)
        annotation["cls"] = torch.ones((len(boxes),), dtype=torch.int64)
        annotation["img_size"] = [self.size, self.size]
        annotation["image_id"] = torch.tensor([index])
        return image, annotation, image_id

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def load_mosaic_image_and_boxes(self, index):
        s = self.imsize
        yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
        indices = [index] + random.sample(range(len(self.data)), 3)

        result_boxes = []
        for i, index in enumerate(indices):
            img, boxes, (h, w) = self.load_image_and_boxes(index)

            if i == 0:
                result_image = np.full((s * 2, s * 2, 3), 114 / 255.0, dtype=np.float32)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            # img4[ymin:ymax, xmin:xmax]
            result_image[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.vstack(result_boxes)
        result_boxes[:, 0:] = np.clip(result_boxes[:, 0:], 0, 2 * s).astype(np.int32)
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where(
                (result_boxes[:, 2] - result_boxes[:, 0])
                * (result_boxes[:, 3] - result_boxes[:, 1])
                > 0
            )
        ]
        return result_image, result_boxes

    def load_image_and_boxes(self, index):
        """Load the original Images and Boxes: Images are normalized by 255.0: Boxes (Pascal VOC)"""
        image_id = self.data["image_id"][index]
        image = cv2.imread(str(self.root / f"{image_id}.png"), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        boxes = self.data["bboxes"][index]
        boxes = albu.convert_bboxes_from_albumentations(
            boxes, "pascal_voc", *image.shape[:2]
        )
        return image, np.array(boxes), image.shape[:2]


class LoadImagesAndLabelsCOCO(Dataset):
    def __init__(self, root, data, transforms, size: int):
        self.root = Path(root)
        self.data = data
        self.tfms = transforms
        self.size = size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = self.data["image_id"][index]
        image_path = self.root / f"{image_id}.png"
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB).astype(
            np.float32
        )
        image /= 255.0
        boxes = self.data["bboxes"][index]
        boxes = albu.convert_bboxes_from_albumentations(
            boxes, "pascal_voc", *image.shape[:2]
        )
        boxes = np.array(boxes)
        labels = np.ones((boxes.shape[0],), dtype=np.int64)

        sample = self.tfms(image=image, bboxes=boxes, labels=labels.tolist())
        image = sample["image"]
        boxes = sample["bboxes"]

        num_objs = len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["img_size"] = [self.size, self.size]
        return image, target

    @staticmethod
    def collate_fn(batch):
        return utils.collate_fn(batch)


@torch.no_grad()
def evaluate(
    model,
    data_loader,
    coco_dataloader,
    device,
    score_threshold=0.001,
    amp_autocast=suppress,
    print_freq=50,
    log_suffix=" ",
):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test " + log_suffix + " : "

    coco = coco_utils.get_coco_api_from_dataset(coco_dataloader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = coco_eval.CocoEvaluator(coco, iou_types)

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        image, ann, _ = batch
        images = torch.stack(image)
        bs = images.shape[0]
        inputs = images.to(device, non_blocking=True).float()

        target = {}
        target["bbox"] = [a["bbox"].to(device, non_blocking=True) for a in ann]
        target["cls"] = [a["cls"].to(device, non_blocking=True) for a in ann]
        target["img_scale"] = (
            torch.tensor([1] * bs).float().to(device, non_blocking=True)
        )
        target["img_size"] = (
            torch.tensor([a["img_size"] for a in ann]).to(device).float()
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with amp_autocast():
            m_outputs = model(inputs, target)

        detections = []
        for i in range(bs):
            boxes = m_outputs["detections"][i][:, :4].data.cpu().numpy()
            scores = m_outputs["detections"][i][:, 4].data.cpu().numpy()
            classes = m_outputs["detections"][i][:, 5].data.cpu().numpy()
            indexes = np.where(scores > score_threshold)[0]
            boxes, scores, classes = boxes[indexes], scores[indexes], classes[indexes]
            detections.append(
                {
                    "boxes": torch.as_tensor(boxes),
                    "scores": torch.as_tensor(scores),
                    "labels": torch.as_tensor(classes),
                }
            )

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in detections]
        model_time = time.time() - model_time

        res = {
            target["image_id"].item(): output for target, output in zip(ann, outputs)
        }
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        m_outputs.pop("detections")
        loss_dict_reduced = utils.reduce_dict(m_outputs)
        metric_logger.update(
            model_time=model_time, evaluator_time=evaluator_time, **loss_dict_reduced
        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    m = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    m["coco"] = coco_evaluator.coco_eval["bbox"].stats[1]
    return m


def train_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch,
    lr_scheduler=None,
    print_freq=50,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    out_dir=None,
):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    for ni, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image, ann, _ = batch
        images = torch.stack(image)
        inputs = images.to(device, non_blocking=True).float()

        target = {}
        target["bbox"] = [a["bbox"].to(device, non_blocking=True) for a in ann]
        target["cls"] = [a["cls"].to(device, non_blocking=True) for a in ann]

        with torch.no_grad():
            if ni < 3 and out_dir:
                box_images = []
                f = out_dir / f"train_batch{ni}.jpg"
                for im, box in zip(inputs, target["bbox"]):
                    box = box.data.cpu().numpy()
                    im = im.data.cpu().numpy().transpose(1, 2, 0) * 255
                    im = np.ascontiguousarray(im, dtype=np.uint8)
                    tl = round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1
                    for x in box:
                        c1, c2 = (int(x[1]), int(x[0])), (int(x[3]), int(x[2]))
                        cv2.rectangle(im, c1, c2, (255, 0, 0), tl, cv2.LINE_AA)
                    im = im.astype(float) / 255.0
                    im = torch.from_numpy(im.transpose(2, 0, 1)).float()
                    box_images.append(im)
                box_images = torch.stack(box_images)
                box_images = torchvision.utils.make_grid(
                    box_images, nrow=4, normalize=True, padding=5
                )
                torchvision.utils.save_image(box_images, f)

        with amp_autocast():
            loss_dict = model(inputs, target)

        losses = loss_dict["loss"]

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_value = loss_dict_reduced["loss"].item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                losses, optimizer, clip_grad=10.0, parameters=model.parameters()
            )
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(**loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# fmt: off
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    # Input
    parser.add_argument("--fold", default=0, type=int, help="Fold number (default: 0)")
    parser.add_argument("--data", default="./grouped_df.csv", type=str, 
                        help="Path to the Annotations csv",)
    parser.add_argument("--root", default="../mmdetection/train/", type=str, 
                        help="Data root")
    
    # Output
    parser.add_argument("--name", default='cadd1bd', type=str, 
                        help="Run Name")
    parser.add_argument("--print-every", default=50, type=int, 
                        help="Print Interval (default: 50)")
    
    
    # Augmentations/ Training Parameters
    parser.add_argument("--device", default="cuda", 
                        help="device (default: cuda)")
    parser.add_argument("--bs", default=5, type=int, 
                        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=40, 
                        help="Total Number of Training Epochs (default: 40)", type=int,)
    parser.add_argument("--mosaic", default=0.5,type=float,
                        help="Mosaic Augmentation Prob (default: 0.5)",)
    parser.add_argument("--mixup", default=0.5, type=float,
                        help="MixUp Augmentation Prob (default: 0.5)",)
    parser.add_argument("--set_bn_eval",action="store_true",default=False,
                        help="Set BN layers to eval mode",)
    parser.add_argument("--lr", default=5e-04, type=float, 
                        help="Learning Rate (default: 5e-04)")
    parser.add_argument("--output",default="",type=str,metavar="PATH",
                        help="path to output folder (default: none, current dir)")

    # Model Args
    parser.add_argument("--model",default="tf_efficientdet_d1",type=str,metavar="MODEL",
                        help='Name of model to train (default: "tf_efficientdet_d5_ap"')
    parser.add_argument("--initial_checkpoint",default="",type=str,metavar="PATH",
                        help="Initialize model from this checkpoint (default: none)")
    parser.add_argument("--image-size", default=512, type=int, help="Image Size size % 128 = 0 (default: 512)")

    # Model Exponential Moving Average
    parser.add_argument("--model-ema",action="store_true",default=False,
                        help="Enable tracking moving average of model weights",)
    parser.add_argument("--model-ema-decay",type=float,default=0.9998,
                        help="decay factor for model weights moving average (default: 0.9998)")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    return parser
# fmt:on


def main(args, args_text):
    setup_default_logging()
    seed_everything(42)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    print(f"Device Initialized {device}")

    # Data loading code
    print("Loading data")
    marking = pd.read_csv(args.data)
    marking.bboxes = marking.bboxes.map(lambda x: ast.literal_eval(x))
    marking = marking.loc[marking.label == "opacity"]

    df_train = marking.query("kfold!=@args.fold").reset_index(drop=True, inplace=False)
    df_valid = marking.query("kfold==@args.fold").reset_index(drop=True, inplace=False)

    print("-- no. images: train - {}, valid - {}".format(len(df_train), len(df_valid)))

    train_presets, valid_presets = get_augmentations(args.image_size)

    print(f"-- image size: {args.image_size} x {args.image_size}")

    print("albumentations: " + ", ".join(f"{x}" for x in train_presets.transforms))

    ds_train = LoadImagesAndLabels(
        args.root,
        df_train,
        transforms=train_presets,
        imsize=args.image_size,
        mixup=args.mixup,
        mosaic=args.mosaic,
        size=args.image_size,
        test=False,
    )

    ds_valid = LoadImagesAndLabels(
        args.root,
        df_valid,
        transforms=valid_presets,
        size=args.image_size,
        test=True,
        imsize=args.image_size,
        mosaic=0,
        mixup=0,
    )
    ds_valid_coco = LoadImagesAndLabelsCOCO(
        args.root, df_valid, transforms=valid_presets, size=args.image_size
    )

    print("Creating data loaders")
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            ds_train, shuffle=True, seed=42
        )
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            ds_valid, shuffle=False, seed=42
        )
        valid_coco_sampler = torch.utils.data.distributed.DistributedSampler(
            ds_valid_coco, shuffle=False, seed=42
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(ds_train)
        valid_sampler = torch.utils.data.SequentialSampler(ds_valid)
        valid_coco_sampler = torch.utils.data.SequentialSampler(ds_valid_coco)

    train_dataloader = DataLoader(
        dataset=ds_train,
        batch_size=args.bs,
        sampler=train_sampler,
        collate_fn=ds_train.collate_fn,
    )

    valid_dataloader = DataLoader(
        dataset=ds_valid,
        batch_size=args.bs,
        collate_fn=ds_valid.collate_fn,
        sampler=valid_sampler,
    )

    valid_dataloader_coco = DataLoader(
        dataset=ds_valid_coco,
        batch_size=args.bs,
        collate_fn=ds_valid_coco.collate_fn,
        sampler=valid_coco_sampler,
    )

    print("Creating model")
    model = create_model(
        model_name=args.model,
        bench_task="train",
        num_classes=1,
        pretrained=True,
        bench_labeler=True,
        pretrained_backbone=True,
        image_size=[args.image_size, args.image_size],
    )

    nps = sum([m.numel() for m in model.parameters()])
    print("Model %s created, param count: %d" % (args.model, nps))

    if args.initial_checkpoint:
        print("loading model from %s" % args.initial_checkpoint)
        load_checkpoint(unwrap_bench(model), args.initial_checkpoint, use_ema=True)

    if args.set_bn_eval:
        set_bn_eval(model)
        print(f"BatchNorm Layers set to eval...")

    model.to(device)

    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(model, decay=args.model_ema_decay)
        print("Using ModelEma for training ...")
    else:
        model_ema = None

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    params = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    gp = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(gp, lr=args.lr)

    ls = len(train_dataloader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ls)

    print("Using native Torch AMP. Training in mixed precision.")
    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()

    saver = None

    if utils.is_main_process():
        output_dir = ""
        output_base = args.output if args.output else "./output"
        output_dir = get_outdir(output_base, args.name, f"fold_{args.fold}")
        output_dir = Path(output_dir)

        print(f"-- output: {output_dir}")

        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

        saver = CheckpointSaver(
            model,
            optimizer,
            model_ema=model_ema,
            checkpoint_dir=output_dir,
            max_history=1,
            amp_scaler=loss_scaler,
            decreasing=False,
            unwrap_fn=unwrap_bench,
        )

    print(f"-- scheduled epochs : {args.epochs}")
    start_time = time.time()
    best_metric = None
    best_epoch = None

    # nbs = 64  # nominal batch size
    # accumulate = max(round(nbs / args.bs), 1)  # accumulate loss before optimizing

    # print(f"-- accumulate : {accumulate}")

    for epoch in range(args.epochs):
        train_stats = train_epoch(
            model,
            optimizer,
            train_dataloader,
            device,
            epoch=epoch,
            amp_autocast=amp_autocast,
            loss_scaler=loss_scaler,
            print_freq=args.print_every,
            lr_scheduler=scheduler,
            model_ema=model_ema,
            out_dir=output_dir,
        )

        # evaluate after every epoch
        eval_stats = evaluate(
            model,
            valid_dataloader,
            valid_dataloader_coco,
            device=device,
            amp_autocast=amp_autocast,
            print_freq=args.print_every,
        )

        if model_ema is not None:
            # Evaluate with EMA weights
            ema_stats = evaluate(
                model_ema.module,
                valid_dataloader,
                valid_dataloader_coco,
                device=device,
                amp_autocast=amp_autocast,
                print_freq=args.print_every,
                log_suffix=" (EMA)",
            )
        else:
            ema_stats = None

        if ema_stats is not None:
            log_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in eval_stats.items()},
                **{f"ema_{k}": v for k, v in ema_stats.items()},
            }
        else:
            log_stats = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in eval_stats.items()},
            }

        if utils.is_main_process():
            with (output_dir / "metrics.json").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if saver is not None:
            if ema_stats is not None:
                eval_stats = ema_stats
            best_metric, best_epoch = saver.save_checkpoint(
                epoch=epoch, metric=eval_stats["coco"]
            )
            print("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    if best_metric is not None:
        print("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


if __name__ == "__main__":
    import argparse

    import yaml

    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = argparse.ArgumentParser(
        description="Training Config", add_help=False
    )
    config_parser.add_argument(
        "-c",
        "--config",
        default="",
        type=str,
        metavar="FILE",
        help="YAML config file specifying default arguments",
    )

    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()

    parser = get_args_parser()
    if args_config.config:
        with open(args_config.config, "r") as f:
            config_args = yaml.safe_load(f)
            # Override defaults with config file values
            parser.set_defaults(**config_args)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    main(args, args_text)
