import os
import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from lightning import LightningModule
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from evaluations.coco_eval import CocoEvaluator
from torch.utils.data import DataLoader
from evaluations.utils import collate_fn
from evaluations.coco_utils import get_coco_api_from_dataset
from datasets.mrcnndataset import MRCNNDataset
from augmentation.augmentation import (
    get_train_transform,
    get_valid_transform
)
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class MRCNNModule(LightningModule):
    IOU_TYPES = ['segm']

    def __init__(self,
                 num_classes: int,
                 learning_rate: float,
                 data_path: str,
                 batch_size: int,
                 workers: int,
                 input_size: int,
                 num_channels: int=3
        ) -> None:
        super(MRCNNModule, self).__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.learning_rate = learning_rate
        self.data_path = data_path
        self.batch_size = batch_size
        self.workers = workers
        self.input_size = input_size
        self.model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            self.model.roi_heads.box_predictor.cls_score.in_features,
            self.num_classes
        )
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            self.model.roi_heads.mask_predictor.conv5_mask.in_channels,
            hidden_layer,
            self.num_classes
        )
        self.add_channels()
        self.coco_evaluator = None
        self.coco_dataset = None
        self.train_step_losses = list()
        self.train_step_mask_losses = list()
        self.val_step_losses = list()
        self.val_step_mask_losses = list()

    def add_channels(self) -> None:
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        for _ in range(self.num_channels-3):
            image_mean.append(image_mean[0])
            image_std.append(image_std[0])
        conv1_w = self.model.backbone.body.conv1.weight.clone()
        conv1_w_idx0 = self.model.backbone.body.conv1.weight[:, 0].clone()
        self.model.transform = GeneralizedRCNNTransform(
            min_size=800, max_size=1333, image_mean=image_mean, image_std=image_std
        )
        self.model.backbone.body.conv1 = nn.Conv2d(
            in_channels=self.num_channels, out_channels=64, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        with torch.no_grad():
            self.model.backbone.body.conv1.weight[:, :3].weight = conv1_w
            for idx in range(3, self.num_channels):
                self.model.backbone.body.conv1.weight[:, idx].weight = conv1_w_idx0

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        return y

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def training_step(self, batch: Tensor, *batch_idx: int) -> dict[str, Tensor]:
        images, targets = batch
        images = [image for image in images]
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        loss = sum(loss for loss in outputs.values())
        loss_mask = outputs['loss_mask']
        self.log_dict({'loss_mask': loss_mask})
        self.train_step_losses.append(loss.detach())
        self.train_step_mask_losses.append(loss_mask.detach())
        return {'loss': loss, 'loss_mask': loss_mask.detach()}

    def on_train_epoch_end(self) -> None:
        mean_loss = torch.stack(self.train_step_losses).mean()
        mean_loss_mask = torch.stack(self.train_step_mask_losses).mean()
        mean_losses = {'mean_loss': mean_loss, 'mean_loss_mask': mean_loss_mask}
        self.log_dict(mean_losses)
        self.train_step_losses.clear()
        self.train_step_mask_losses.clear()

    def validation_step(self, batch: Tensor, *batch_idx: int) -> None:
        images, targets = batch
        images = [image for image in images]
        outputs = self.model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

    def on_validation_epoch_end(self) -> None:
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        segm_prec_iou = self.coco_evaluator.coco_eval['segm'].stats[0]
        self.coco_evaluator = CocoEvaluator(self.coco_dataset, self.IOU_TYPES)
        metrics = {'segm_prec_iou': segm_prec_iou}
        self.log_dict(metrics)

    def setup(self, stage: str) -> None:
        val_dataset = MRCNNDataset(
            root=os.path.join(self.data_path, 'validation'),
            num_channels=self.num_channels,
            transforms=get_valid_transform(size=self.input_size)
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            collate_fn=collate_fn
        )
        self.coco_dataset = get_coco_api_from_dataset(self.val_loader.dataset)
        self.coco_evaluator = CocoEvaluator(self.coco_dataset, self.IOU_TYPES)
        train_dataset = MRCNNDataset(
            root=os.path.join(self.data_path, 'train'),
            num_channels=self.num_channels,
            transforms=get_train_transform(size=self.input_size)
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True, num_workers=self.workers,
            collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
