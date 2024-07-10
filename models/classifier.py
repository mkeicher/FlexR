import lightning as L
from torchmetrics import AUROC, MetricCollection, F1Score, Accuracy
import torch
import torch.nn as nn
from transformers.file_utils import ModelOutput
from utils.lr_scheduler import CosineWarmupScheduler
from pathlib import Path
import pandas as pd
from torchvision.models import resnet50
from monai.networks.nets import DenseNet121, DenseNet169
from transformers import AutoModelForImageClassification
import torch


class ClassifierModule(L.LightningModule):
    def __init__(
        self,
        num_classes,
        in_channels=3,
        lr_scheduler='cosine', 
        learning_rate=1e-4,
        min_lr=1e-8, 
        warmup_epochs=1,
        pretrained=True,
        image_encoder='densenet121',
        pretrained_densenet = None,
        task='multilabel',
        target_name='attributes',
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.image_encoder == 'resnet50':
             self.image_encoder = resnet50(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        elif self.hparams.image_encoder == 'densenet121':
            self.image_encoder = DenseNet121(
                pretrained=self.hparams.pretrained, 
                in_channels=self.hparams.in_channels, 
                out_channels=self.hparams.num_classes,
                spatial_dims=2,
            )
            if self.hparams.pretrained_densenet is not None:
                # checkpoint = "artifacts/mimic-classifier-paper/densenet121_stellar-sweep-9_epoch20_AUC79.chkpt"
                state_dict = torch.load(Path(self.hparams.pretrained_densenet))
                state_dict = {k.replace('image_encoder.', ''):v for k,v in state_dict.items() if 'class_layers' not in k}
                self.image_encoder.load_state_dict(state_dict, strict=False)
        else:
            self.image_encoder = AutoModelForImageClassification.from_pretrained(
                self.hparams.image_encoder, 
                num_labels=self.hparams.num_classes, 
                ignore_mismatched_sizes=True
            )
            if not self.hparams.pretrained:
                 self.image_encoder.init_weights()
        
        # setup loss
        if self.hparams.task in ['binary', 'multilabel']:
            self.loss = nn.BCEWithLogitsLoss()
        elif self.hparams.task == 'multiclass':
            self.loss = nn.CrossEntropyLoss()
        else:
            raise ValueError
        
        # metrics
        metric_collection = MetricCollection({
            'Accuracy': Accuracy(task=self.hparams.task, num_classes=self.hparams.num_classes, num_labels=self.hparams.num_classes),
            'AUROC': AUROC(task=self.hparams.task, num_classes=self.hparams.num_classes, num_labels=self.hparams.num_classes),
            'F1Score': F1Score(task=self.hparams.task, num_classes=self.hparams.num_classes, num_labels=self.hparams.num_classes),
        })
    
        self.train_metrics = metric_collection.clone(prefix=f'train/')
        self.valid_metrics = metric_collection.clone(prefix=f'valid/')
        self.test_metrics = metric_collection.clone(prefix=f'test/')


    def forward(self, batch):
        outputs = self.image_encoder(batch.image)
        if isinstance(outputs, ModelOutput):
            return outputs.logits
        else:
            return outputs
    
    def step(self, batch, phase):
        if len(batch.study_id) != len(batch.image):
            batch.image = batch.image[:len(batch.study_id)]
        
        logits = self(batch)
        proba = logits.softmax() if self.hparams.task == 'multiclass' else logits.sigmoid()
        target = batch[self.hparams.target_name]

        loss = self.loss(logits, target.float())
        getattr(self, f'{phase}_metrics')(proba, target)
        self.log_dict(getattr(self, f'{phase}_metrics'), on_step=False, on_epoch=True)
        self.log(f'{phase}/loss', loss, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')

    def configure_optimizers(self):
        params = self.image_encoder.parameters()

        optimizer = torch.optim.Adam(params, lr=self.hparams.learning_rate)

        steps_per_epoch = len(self.trainer.datamodule.train_dataloader()) // self.trainer.accumulate_grad_batches
        total_steps = self.trainer.max_epochs * steps_per_epoch

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": CosineWarmupScheduler(
                    optimizer, 
                    warmup=int(self.hparams.warmup_epochs * steps_per_epoch), 
                    max_iters=total_steps, 
                    min_lr=self.hparams.min_lr
                ),
                "interval": "step",
                "frequency": 1,
            }
        }