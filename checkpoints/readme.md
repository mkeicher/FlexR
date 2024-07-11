[![arxiv](https://img.shields.io/badge/arXiv-2203.15723-b31b1b)](https://arxiv.org/abs/2203.15723)
[![MIDL 2023](https://img.shields.io/badge/OpenReview-MIDL_2023-8c1b13)](https://openreview.net/forum?id=wiN5LQThnIV)
[![MIDL 2023](https://img.shields.io/badge/MIDL-2023-b18630)](https://2023.midl.io/papers/p162)
[![PLMR](https://img.shields.io/badge/PLMR-2024-0c236b)](https://proceedings.mlr.press/v227/keicher24a.html)
# FlexR: Few-shot Classification with Language Embeddings for Structured Reporting of Chest X-rays
Required checkpoints:
- `densenet121_pretrained.ckpt`
- `FlexR-CLIP.ckpt`

Run the following commands or manually download the weights from the release page.
## Download the weights in this directory

### 1) Pretrained DenseNet121 for image encoder initialization of FlexR-CLIP
```
wget -P checkpoints https://github.com/mkeicher/FlexR/releases/download/v1.0/densenet121_pretrained.ckpt
```

### 2) FlexR-CLIP model weights
```
wget -P checkpoints https://github.com/mkeicher/FlexR/releases/download/v1.0/FlexR-CLIP.ckpt
```