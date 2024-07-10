
from torchvision import transforms as T
from pathlib import Path 
import lightning as L
import pandas as pd
import random
import torch
import numpy as np
from transformers import BatchFeature
from torchvision.transforms import InterpolationMode
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import NDArrayDecoder, IntDecoder
from ffcv.transforms import ToTensor

from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer
import warnings

from data.sampler import ImbalancedDatasetSampler, CustomFfcvLoader

warnings.simplefilter('once')

class DenormalizedJitter(T.ColorJitter):
    def __init__(self, brightness=0.1, contrast=0.2, saturation=0.2, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
    def forward(self, x):
        return super(DenormalizedJitter, self).forward((x+1)/2.)*2.-1

class ImagenomeDataModule(L.LightningDataModule):
    def __init__(self, 
            batch_size=256, scale_min=0.75, num_workers=12, in_channels=3,
            data_dir="/mnt/polyaxon/data1/mimic/datasets/220209_mimic-cxr_imagenome",
            image_size=224, outputs=['study_id', 'image', 'tokens'],
            max_token_length=128, tokenizer='allenai/scibert_scivocab_uncased', 
            random_sentence=5, add_triplets_to_sentences=0,
            augment_rotation=15, augment_jitter=1, 
            mimic_cxr_jpg_split_csv='/home/data/DIVA/mimic/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv',
            exclude_mimic_test_split=True,
            mask_triplets_without_annotations=True,
            sampling = None,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = Path(self.hparams.data_dir)
        
        # load mimic-cxr-jpg test split study ids to be excluded in training and validation (not testing)
        if self.hparams.exclude_mimic_test_split:
            self.mimic_test_split = pd.read_csv(self.hparams.mimic_cxr_jpg_split_csv).query('split == "test"').study_id.reset_index(drop=True)
        else:
            self.mimic_test_split = []

        # load imagenome features - study_id, sentences, triplets, triplets_one_hot
        self.imagenome_features = pd.read_hdf(self.data_dir / 'imagenome_features.h5')
        self.imagenome_attributes = pd.read_hdf(self.data_dir / 'imagenome_attributes.h5')

        #  filter triplets that have no positive annotations
        self.triplet_mask = np.array(self.imagenome_features.triplets_one_hot.tolist()).sum(0) > 0

        # ImaGenome triplets
        self.attributes = ['lung opacity','pleural effusion','atelectasis','enlarged cardiac silhouette','pulmonary edema/hazy opacity'
                ,'pneumothorax','consolidation','fluid overload/heart failure','pneumonia']
        self.locations = ['right lung','right apical zone','right upper lung zone','right mid lung zone','right lower lung zone','right hilar structures'
                ,'left lung','left apical zone','left upper lung zone','left mid lung zone','left lower lung zone','left hilar structures'
                ,'right costophrenic angle','left costophrenic angle','mediastinum','upper mediastinum','cardiac silhouette','trachea']
        relation = 'in the'
        self.triplets = [f'{a} {relation} {l}' for a in self.attributes for l in self.locations]
        
        self.long_report_length = []

        self.pipelines = {
            'train' : {
                'image':  
                    # [NDArrayDecoder(), ToTensor(), T.Resize(size=self.hparams.image_size)],
                    [NDArrayDecoder(), ToTensor()] +
                    ([T.RandomRotation(self.hparams.augment_rotation, InterpolationMode.BILINEAR, expand=False, fill=-1.)] if self.hparams.augment_rotation else []) +
                    [T.RandomResizedCrop(self.hparams.image_size, scale=(self.hparams.scale_min, 1.), ratio=(1., 1.))] +
                    ([DenormalizedJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0)] if self.hparams.augment_jitter else []),
                'study_id': [IntDecoder()]
            },
            'test': {
                'image':  [
                    NDArrayDecoder(), ToTensor(),
                ],
                'study_id': [IntDecoder()]
            }
        }

        if self.hparams.tokenizer == 'ratchet':
                self.tokenizer = ByteLevelBPETokenizer(
                    'preprocess/mimic-vocab.json',
                    'preprocess/mimic-merges.txt',
                )
                # self.tokenizer.enable_truncation(max_length=self.hparams.max_token_length)
                self.tokenizer.enable_padding(length=512, pad_token='<pad>')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.tokenizer)
    
    def tokenize(self, report):
        if self.hparams.tokenizer == 'ratchet':
            encoded = self.tokenizer.encode_batch(
                report, 
                add_special_tokens=True,
            )
            # Convert to tensors and create a dictionary similar to batch_encode_plus output
            input_ids = torch.tensor([enc.ids for enc in encoded])
            attention_mask = torch.tensor([enc.attention_mask for enc in encoded])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        else:
            return self.tokenizer.batch_encode_plus(report, padding='max_length', truncation=False, max_length=512, return_tensors="pt")

    def get_subject_mask(self, study_id):
        # filter studies with missing imagenome features and mimic test split
        mask = np.ones(len(study_id), dtype=bool)

        # remove studies with missing imagenome features from the batch
        missing_ids = set(study_id) - set(self.imagenome_features.index)
        if missing_ids:    
            warnings.warn(f"Dropping study ids: {missing_ids} (missing in imagenome features)")
            # remove missing study ids from the batch
            mask = np.isin(study_id, list(self.imagenome_features.index)) & mask
        
        # drop studies in mimic test split from training and validation
        if (self.trainer.training or self.trainer.validating) and self.hparams.exclude_mimic_test_split:
            mask = ~np.isin(study_id, self.mimic_test_split) & mask
        
        return mask
        

    def on_before_batch_transfer(self, batch, dataloader_idx):
        image, study_id = batch
        if image.size(1) == 1 and self.hparams.in_channels == 3:
            image = image.repeat([1,3,1,1])
        study_id = np.array(study_id[:,0])
        
        mask = self.get_subject_mask(study_id)
        study_id = study_id[mask]
        image = image[mask]

        # add imagenome trplet sentences if enabled
        if self.hparams.add_triplets_to_sentences:
            report_sentences = (self.imagenome_features.loc[study_id].sentences + self.imagenome_features.loc[study_id].triplets)
        else:
            report_sentences = self.imagenome_features.loc[study_id].sentences

        # random sentence selection if enabled
        if self.hparams.random_sentence and self.trainer.training:
            report = report_sentences.map(lambda x: ' '.join(random.sample(x.split('|'), k=random.randint(1, min(len(x.split('|')), self.hparams.random_sentence))))).tolist()
        else:
            report = report_sentences.map(lambda x: ' '.join(x.split('|'))).tolist()
            # alternative pretraining using full report based on RATCHET preprocessed reports: https://github.com/farrell236/RATCHET/tree/master/preprocessing/mimic
            # report = self.ratchet_mimic_labels.loc[study_id, 'Reports'].tolist()    

        batch = {}
        if 'study_id' in self.hparams.outputs:
            batch['study_id'] = torch.tensor(study_id)
        if 'image' in self.hparams.outputs:
            batch['image'] = image
        if 'report' in self.hparams.outputs:
            batch['report'] = report
        if 'tokens' in self.hparams.outputs:
            batch.update(self.tokenize(report))
            for i, attn in enumerate(batch['attention_mask']):
                # random cropping for reports too long
                if attn.sum() > self.hparams.max_token_length and self.trainer.training:
                    random_start = random.randint(0, attn.sum() - self.hparams.max_token_length)
                    batch['input_ids'][i, :self.hparams.max_token_length] = batch['input_ids'][i, random_start:random_start+self.hparams.max_token_length].clone()
                    batch['attention_mask'][i, :self.hparams.max_token_length] = attn[random_start:random_start+self.hparams.max_token_length].clone()
            batch['input_ids'] = batch['input_ids'][:, :self.hparams.max_token_length]
            batch['attention_mask'] = batch['attention_mask'][:, :self.hparams.max_token_length]

        if 'triplets' in self.hparams.outputs:
            batch['triplets'] = self.imagenome_features.loc[study_id].triplets
        if 'triplets_onehot' in self.hparams.outputs:
            batch['triplets_onehot'] = torch.tensor(np.stack(self.imagenome_features.triplets_one_hot.loc[study_id].values), dtype=torch.int16)
            if self.hparams.mask_triplets_without_annotations:
                batch['triplets_onehot'][:, ~self.triplet_mask] = -1
        if 'attributes' in self.hparams.outputs:
            batch['attributes'] = torch.tensor(np.stack(self.imagenome_attributes.loc[study_id].values), dtype=torch.int16)
        return BatchFeature(batch)
        
        
    def train_dataloader(self):
        if self.hparams.sampling:
            # idxs = np.load(self.data_dir / 'train_idxs.npy')
            temp_loader = Loader(self.data_dir /  'train.beton', batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                order=OrderOption.SEQUENTIAL, pipelines={'study_id': [IntDecoder()]})
            subject_ids = []
            for batch in temp_loader:
                subject_ids.extend(batch[1].flatten().tolist())
            valid_idxs = np.arange(len(subject_ids))[np.isin(subject_ids, self.imagenome_features.index)]
            valid_subject_ids = np.array(subject_ids)[valid_idxs]
            labels = np.stack(self.imagenome_features.loc[valid_subject_ids].triplets_one_hot.values)
            imbalanced_train_sampler = ImbalancedDatasetSampler(sampling_factor=self.hparams.sampling, labels=labels, seed=None, shuffle=True)
            # batch_size = 1 if ('fewshot' in self.hparams.sampling_factor) else self.hparams.batch_size
            batch_size = self.hparams.batch_size
            return CustomFfcvLoader(imbalanced_train_sampler, self.data_dir / 'train.beton', batch_size=batch_size, num_workers=self.hparams.num_workers,
                    order=OrderOption.SEQUENTIAL, pipelines=self.pipelines['train'])
        else:
            return Loader(self.data_dir / 'train.beton', batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                order=OrderOption.RANDOM, pipelines=self.pipelines['train'])
    
    def val_dataloader(self):
        return Loader(self.data_dir / 'valid.beton', batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                order=OrderOption.SEQUENTIAL, pipelines=self.pipelines['test'])

    def test_dataloader(self):
        return Loader(self.data_dir / 'test.beton', batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers,
                order=OrderOption.SEQUENTIAL, pipelines=self.pipelines['test'])