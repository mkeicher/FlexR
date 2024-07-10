import torch
import torch.nn as nn
import lightning as L
from utils.lr_scheduler import CosineWarmupScheduler
from transformers.file_utils import ModelOutput
from transformers import CLIPModel, AutoModel, AutoTokenizer, AutoConfig, BatchFeature
from transformers.models.clip.modeling_clip import CLIPOutput
from pathlib import Path
from monai.networks.nets import DenseNet121
from transformers.file_utils import ModelOutput
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F

# from https://github.com/openai/CLIP/blob/main/clip/model.py
class AttentionPool2d(nn.Module):
    def __init__(self, spatial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spatial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        std = self.c_proj.in_features ** -0.5
        nn.init.normal_(self.q_proj.weight, std=std)
        nn.init.normal_(self.k_proj.weight, std=std)
        nn.init.normal_(self.v_proj.weight, std=std)
        nn.init.normal_(self.c_proj.weight, std=std)

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return ModelOutput(logits=None, pooler_output=x.squeeze(0))

class PoolOutput(nn.Module):
    def forward(self, x):
        return ModelOutput(logits=None, pooler_output=x)

class ModifiedDenseNet121(DenseNet121):
    def __init__(self, in_channels, use_attention_pool):
        super().__init__(pretrained=True, in_channels=in_channels, out_channels=1, spatial_dims=2)

        self.embed_dim = self.class_layers.out.in_features
        if use_attention_pool:
            self.class_layers = AttentionPool2d(spatial_dim=7, embed_dim=self.embed_dim, num_heads=8, output_dim=self.embed_dim)
        else:
            self.class_layers.out = PoolOutput()
    
    def forward(self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return super().forward(pixel_values)

def calculate_cosine_similarity(embedding_1, embedding_2):
    # Normalize the vectors to a length of one and then perform a matrix multiplication (dot product)
    return F.normalize(embedding_1) @ F.normalize(embedding_2).T

def clip_loss(similarities, temperature):
    # Calcular the CLIP loss as the average of the text and image losses matching pairs in the batch
    batch_size = similarities.size(0)
    logits = similarities * temperature.exp()
    labels = torch.arange(batch_size, device=logits.device)
    text_loss = F.cross_entropy(logits, labels)
    image_loss = F.cross_entropy(logits.T, labels)
    return (text_loss + image_loss) / 2.0

def siglip_loss(similarities, temperature, bias):
    # Calculate the loss for the Siglip loss function following https://arxiv.org/abs/2303.15343
    batch_size = similarities.size(0)
    labels = 2 * torch.eye(batch_size, device=similarities.device) - 1 # 1s on the diagonal, -1s everywhere else
    logits = similarities * temperature.exp() + bias
    loss = -F.logsigmoid(labels * logits).sum() / batch_size
    return loss

class FlexrClipModule(L.LightningModule):
    def __init__(self,
            image_encoder='openai/clip-vit-base-patch16', text_encoder='openai/clip-vit-base-patch16', 
            learning_rate=5e-6, min_lr=1e-8, warmup_epochs=1., clip_model='openai/clip-vit-base-patch16', 
            clip_pretrained=1, freeze='nothing', in_channels=3, tokenizer='ratchet', max_token_length=32, 
            lr_scheduler='cosine_decay', loss='clip', **kwargs
        ):
        super().__init__()
        # save hparams if not done by parent class
        if type(self) is FlexrClipModule:
            self.save_hyperparameters()
        self.clip = CLIPModel.from_pretrained(self.hparams.clip_model)

        if self.hparams.in_channels == 1:
            em = self.clip.vision_model.embeddings
            self.clip.vision_model.embeddings.patch_embedding = nn.Conv2d(
                in_channels=1, out_channels=em.embed_dim, kernel_size=em.patch_size, stride=em.patch_size, bias=False
            )
        if 'clip' not in self.hparams.image_encoder:
            if 'densenet' in self.hparams.image_encoder:
                self.clip.vision_model = ModifiedDenseNet121(in_channels=self.hparams.in_channels, use_attention_pool='attn_pool' in self.hparams.image_encoder)
                if 'densenet121_pretrained' in self.hparams.image_encoder:
                    state_dict = torch.load(self.hparams.image_encoder)['state_dict']
                    state_dict = {k.replace('image_encoder.', ''):v for k,v in state_dict.items()}
                    missing_keys, unexpteded_keys = self.clip.vision_model.load_state_dict(state_dict, strict=False)
                    if len(missing_keys) > 0:
                        raise ValueError(f'Missing keys: {missing_keys}')
                self.clip.visual_projection = nn.Linear(self.clip.vision_model.embed_dim, self.clip.projection_dim, bias=False)
                nn.init.normal_(
                    self.clip.visual_projection.weight,
                    std=self.clip.vision_model.embed_dim ** -0.5 *  1,
                )
            else:
                self.clip.vision_model = AutoModel.from_pretrained(self.hparams.image_encoder)
        if 'clip' not in self.hparams.text_encoder:
            self.clip.text_model = AutoModel.from_pretrained(self.hparams.text_encoder)
            # self.clip.text_model = AutoModel.from_config(AutoConfig.from_pretrained(self.hparams.text_encoder))
            self.clip.text_projection = nn.Linear(self.clip.text_model.config.hidden_size, self.clip.projection_dim, bias=False)
            if 'text_encoder' in self.hparams.freeze:
                for param in self.clip.text_model.parameters():
                    param.requires_grad = False
            if 'image_encoder' in self.hparams.freeze:
                for param in self.clip.vision_model.parameters():
                    param.requires_grad = False        
        if not self.hparams.clip_pretrained:
            self.clip.init_weights()
    
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)
        
        if self.hparams.loss == 'siglip':
            # init following siglip paper https://arxiv.org/abs/2303.15343
            bias_init = -10.
            temperature_init = 10.
            self.clip.logit_bias = nn.Parameter(torch.tensor(bias_init))
            self.clip.logit_scale = nn.Parameter(torch.tensor(temperature_init).log())

    def tokenize(self, text):
        return self.tokenizer.batch_encode_plus(text, padding=True, truncation=True, max_length=self.hparams.max_token_length, return_tensors="pt", return_token_type_ids=False)
    
    def encode_text(self, text):
        # tokenize text
        tokens = self.tokenize(text)
        # move to device of clip model
        tokens = {k: v.to(self.clip.device) for k, v in tokens.items()}
        # get text embeddings
        return self.clip.get_text_features(**tokens)

    def forward(self, pixel_values=None, report=None, input_ids=None, attention_mask=None):
        if input_ids is None:
            tokens = self.tokenize(report)
            input_ids = tokens.input_ids
            attention_mask = tokens.attention_mask

        return self.clip(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_loss=True,
            return_dict=True,
        )

    def step(self, batch):
        clip_output = self(pixel_values=batch.image, input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        if self.hparams.loss == 'siglip':
            image_embeds = clip_output.image_embeds
            text_embeds = clip_output.text_embeds
            similarity = calculate_cosine_similarity(image_embeds, text_embeds)
            loss = siglip_loss(similarity, temperature=self.clip.logit_scale, bias=self.clip.logit_bias)
            return CLIPOutput(
                loss=loss,
                logits_per_image=clip_output.logits_per_image,
                logits_per_text=clip_output.logits_per_text,
                text_embeds=text_embeds,
                image_embeds=image_embeds,
            )
        else: 
            return clip_output
        
    def training_step(self, batch):
        output = self.step(batch)
        self.log('train/loss', output.loss, prog_bar=True)
        self.log('train/accuracy', self.calculate_accuracy(output))
        return output.loss

    def validation_step(self, batch):
        output = self.step(batch)
        self.log('valid/loss',  output.loss, prog_bar=True)
        self.log('valid/accuracy', self.calculate_accuracy(output))
        return output.loss

    def calculate_accuracy(self, output):
        with torch.no_grad():
            return sum(
                (logits.argmax(dim=-1) == torch.arange(logits.size(0), device=logits.device)).float().mean()
                for logits in [output.logits_per_image, output.logits_per_text]
            ) / 2

    def configure_optimizers(self):
        layer_norm_bias_params = [p for name, p in self.clip.named_parameters() if 'layer_norm' in name or 'bias' in name]
        other_params = [p for name, p in self.clip.named_parameters() if not ('layer_norm' in name or 'bias' in name)]
        optimizer = torch.optim.AdamW([
            {'name': 'layer_norm_bias','params': layer_norm_bias_params, 'weight_decay':0},
            {'name': 'other', 'params': other_params, 'weight_decay':0.1},
        ], lr=self.hparams.learning_rate, betas=(0.9,0.98), eps=1e-6)
        steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
        total_steps = self.trainer.max_epochs * steps_per_epoch // self.trainer.accumulate_grad_batches
        
        if self.hparams.lr_scheduler == 'cosine_decay':
            lr_scheduler= { 
                "scheduler": CosineWarmupScheduler(
                    optimizer, 
                    warmup=int(self.hparams.warmup_epochs * steps_per_epoch), 
                    max_iters=total_steps, 
                    min_lr=self.hparams.min_lr
                ),
                "interval": "step",
                "frequency": 1,
            }
        else:
            return optimizer

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.clip.logit_scale.data.clamp_(-torch.tensor(100, device=self.clip.device).log(), torch.tensor(100, device=self.clip.device).log())