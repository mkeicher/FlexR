import numpy as np
import torch
# from pytorch_lightning.utilities import rank_zero_info

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, min_lr=1e-8):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer)
        # rank_zero_info(f'CosineWarmupScheduler initialized with {warmup} warmup steps, max steps {max_iters} and min lr {min_lr}')

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [self.min_lr + base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch > self.max_num_iters and self.max_num_iters > 0:
            epoch = self.max_num_iters
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor