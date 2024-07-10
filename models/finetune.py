import lightning as L
from torchmetrics import AUROC, MetricCollection, F1Score, Accuracy
import torch
import torch.nn as nn
from torch import Tensor
from transformers.file_utils import ModelOutput
from utils.lr_scheduler import CosineWarmupScheduler
from pathlib import Path
import pandas as pd
from torchvision.models import resnet50
from monai.networks.nets import DenseNet121, DenseNet169
from transformers import AutoModelForImageClassification
import torch
from models.clip import FlexrClipModule
from torch.nn import functional as F
from utils.eval import PathologyLocalizationAUC

# original FlexR-1 implementation:
# class PromptSimilarity(nn.Linear):
#     def __init__(self,
#         prompt_embeddings,
#     ) -> None:
#         out_features, in_features = prompt_embeddings.shape
#         super().__init__(in_features=in_features, out_features=out_features, bias=False)
#         self.weight.data = F.normalize(prompt_embeddings)

#     def forward(self, image_embedding: Tensor) -> Tensor:
#         # dot product with normalized vectors = cosine similarity
#         x = F.normalize(image_embedding)
#         x = super().forward(x)
#         return xs

class PromptSimilarity(nn.Module):
    def __init__(self, prompt_embeddings, normalized_weights=False, bias=False):
        super().__init__()
        self.weight = nn.Parameter(F.normalize(prompt_embeddings, dim=1))
        self.bias = nn.Parameter(torch.zeros(prompt_embeddings.shape[0])) if bias else None
        
        # normalize weights to enforce cosine similarity
        self.normalized_weights = normalized_weights

    def forward(self, image_embedding: Tensor) -> Tensor:
        # Normalize weights and input
        normalized_weight = F.normalize(self.weight, dim=1)
        normalized_input = F.normalize(image_embedding, dim=1)
        
        # Compute cosine similarity
        return F.linear(normalized_input, normalized_weight if self.normalized_weights else self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'Cosine similarity with trainable (text) embeddings: in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}'

def logsumexp_sign_loss(similarity, target, gamma=50, ignore_index=-1):
    # FLEXR loss function for finetuning text embeddings
    # see https://arxiv.org/pdf/2203.15723

    # mask targets == -1
    mask = (target != ignore_index).any(0)
    similarity = similarity[:, mask]
    target = target[:, mask]

    # convert targets to similarities i.e. 1 for positive, -1 for negative pair
    target = 2 * target - 1

    # L_{LSES} = log(1 + exp(- y * gamma * similarity))
    return torch.logsumexp(F.pad(-target* gamma * similarity, (0,1)), -1).mean()


class VisionInput(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return self.model(pixel_values)

class FlexrFinetuneModule(FlexrClipModule):
    def __init__(self, prompts=None, task='multilabel', params='all', clip_checkpoint=None, **kwargs):
        # load pretrained clip model if specified
        if clip_checkpoint is not None:
            checkpoint = torch.load(clip_checkpoint)
            hparams = checkpoint['hyper_parameters']
            hparams['learning_rate'] = kwargs.get('learning_rate', hparams['learning_rate'])
            self.save_hyperparameters({**hparams, 'task': task, 'prompts': prompts, 'clip_checkpoint': clip_checkpoint, 'params': params})
            super().__init__(**hparams)
            self.load_state_dict(checkpoint['state_dict'])
            print(f'Loaded CLIP model from {clip_checkpoint}')
        else:
            self.save_hyperparameters()
            super().__init__(**kwargs)
        
        # initialize classifier with text embeddings
        with torch.no_grad():
            self.eval()
            prompt_embeddings = self.encode_text(prompts)
        self.classifier = PromptSimilarity(prompt_embeddings)
        num_classes = len(prompts)

        metric_collection = MetricCollection(PathologyLocalizationAUC())
    
        self.train_metrics = metric_collection.clone(prefix=f'train/')
        self.valid_metrics = metric_collection.clone(prefix=f'valid/')
        self.test_metrics = metric_collection.clone(prefix=f'test/')

        self.temp_running_targets = []

    def forward(self, batch):
        image_embedding = self.clip.get_image_features(batch.image)
        logits = self.classifier(image_embedding)
        return logits

    def step(self, batch, phase):
        logits = self(batch)
        target = batch.triplets_onehot
        loss = logsumexp_sign_loss(logits, target)
        proba = logits.softmax() if self.hparams.task == 'multiclass' else logits.sigmoid()
        if phase != 'train':
            getattr(self, f'{phase}_metrics').update(proba, target)
        self.log(f'{phase}/loss', loss, prog_bar=True)
        
        return loss

    def training_step(self, batch):
        return self.step(batch, 'train')

    def validation_step(self, batch):
        return self.step(batch, 'valid')
    
    def test_step(self, batch):
        return self.step(batch, 'test')
    
    def on_validation_epoch_end(self):
        self.log_dict(self.valid_metrics.compute())
        self.valid_metrics.reset()
    
    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        if self.hparams.params == 'text_embeddings':
            params = self.classifier.parameters()
            for p in self.clip.parameters():
                p.requires_grad = False
        elif self.hparams.params == 'all':
            params = self.parameters()
        else:
            raise ValueError
        
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=0)
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

    # @staticmethod
    # def add_argparse_args(parent_parser):
    #     parser = parent_parser.add_argument_group("FlexrModule")
    #     parser.add_argument("--text_encoder", type=str, default='openai/clip-vit-base-patch16')
    #     parser.add_argument("--image_encoder", type=str, default='openai/clip-vit-base-patch16')
    #     parser.add_argument("--clip_checkpoint", type=str, default='artifacts/model-2krxea41:v300/model.ckpt')
    #     parser.add_argument("--optimizer", type=str, default='AdamW')
    #     parser.add_argument("--learning_rate", type=float, default=1.e-3)
    #     parser.add_argument("--lr_scheduler", type=str, default='cosine decay', choices=['cosine warm restart','cosine decay','none'])
    #     parser.add_argument("--min_lr", type=float, default=1.e-8)
    #     parser.add_argument("--warmup_epochs", type=float, default=1.)
    #     parser.add_argument("--lses_gamma", type=int, default=100)
    #     parser.add_argument("--prompts", type=str, default='attributes', choices=['localized attributes', 'attributes','heart'])
    #     parser.add_argument("--random_initialization", type=str, default='no', choices=['no', 'all', 'classifier', 'image_encoder'])
    #     return parent_parser