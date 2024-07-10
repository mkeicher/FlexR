import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path

from monai.data import Dataset
from monai.transforms import Compose, LoadImageD, Rotate90D, ScaleIntensityRangeD, NormalizeIntensityD, EnsureTypeD, SelectItemsD, ResizeD, SpatialPadD, EnsureChannelFirstD

from ffcv.writer import DatasetWriter
from ffcv.fields import NDArrayField, IntField

import random
from tqdm.auto import tqdm
import simdjson
import json
from multiprocessing import Pool, cpu_count


def generate_triplets(attrs, locs, relation='in the'):
    """Generate all possible triplets from attributes and locations."""
    triplets = [f'{a} {relation} {l}' for a in attrs for l in locs]
    print('generated', len(triplets), 'triplets, with', len(attrs), 'attributes in', len(locs), 'locations:', triplets)
    return triplets

def healthy_lung_sentence(return_random=False):
    """Generate sentences indicating a healthy lung, optionally returning a random one."""
    attributes = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette', 'pulmonary edema', 'hazy opacity', 
                  'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']
    negations = ['no', 'no evidence of', 'there is no']
    negated_attribute_sentence = [n + ' ' + a for a in attributes for n in negations]
    normal_sentences = ['normal cardiac silhouette', 'lungs are clear', 'heart size is normal', 'clear lungs', 'cardiac silhouette is normal']
    all_sentences = negated_attribute_sentence + normal_sentences
    if return_random:
        return random.choice(all_sentences)
    else:
        return all_sentences
    
# Define attributes and locations for generating triplets
attrs = ['lung opacity', 'pleural effusion', 'atelectasis', 'enlarged cardiac silhouette', 'pulmonary edema/hazy opacity',
        'pneumothorax', 'consolidation', 'fluid overload/heart failure', 'pneumonia']

locs = ['right lung', 'right apical zone', 'right upper lung zone', 'right mid lung zone', 'right lower lung zone', 'right hilar structures',
        'left lung', 'left apical zone', 'left upper lung zone', 'left mid lung zone', 'left lower lung zone', 'left hilar structures',
        'right costophrenic angle', 'left costophrenic angle', 'mediastinum', 'upper mediastinum', 'cardiac silhouette', 'trachea']

# Generate all possible triplets
ALL_TRIPLETS = generate_triplets(attrs, locs)

# Generate sentences indicating a healthy lung
ALL_HEALTHY_TRIPLETS = healthy_lung_sentence()
print('healthy triplets generated: ', ALL_HEALTHY_TRIPLETS)

# Initialize the SIMD JSON parser
parser = simdjson.Parser()

class DictToListD():
    def __init__(self, keys):
        self.keys=keys

    def __call__(self, data_dict):
        return [data_dict[k] for k in self.keys]

class ChestImaGenomeDataset(Dataset):
    def __init__(self,
            mimic_cxr_jpg_path,
            chest_imagenome_path,
            split,
            view_positions=['PA', 'AP'],
            image_size=224,
            excluded_studies = [], #58235663, 50798377, 54168089, 53071062, 56724958, 54231141, 53607029, 52035334, 56469752, 50205027, 55295622, 55414712] # missing images
        ):
        
        splits = ['train', 'test', 'valid']
        assert split in splits, f'split must be one of {splits}'
        # load imagenome metadata for the split
        df = pd.read_csv(Path(chest_imagenome_path) / f'files/chest-imagenome/1.0.0/silver_dataset/splits/{split}.csv')

        # filter excluded studies ids and view positions
        df = df.loc[df.ViewPosition.isin(view_positions) & ~df.study_id.isin(excluded_studies)]
        df['image'] = df.path.apply(lambda p: (Path(mimic_cxr_jpg_path) / p).with_suffix('.jpg'))

        transform = Compose([
                LoadImageD(keys=['image']),
                EnsureChannelFirstD(keys='image'),
                Rotate90D(keys='image', k=3),
                ResizeD(keys='image', spatial_size=image_size, size_mode='longest', anti_aliasing=False),
                SpatialPadD(keys='image', spatial_size=image_size, mode='constant', constant_values=0),
                ScaleIntensityRangeD(keys='image', a_min=0, a_max=254, b_min=0., b_max=1., clip=True),
                NormalizeIntensityD(keys='image', subtrahend=0.5, divisor=0.5),
                EnsureTypeD(
                    keys=['image', 'study_id'],
                    data_type='numpy',
                    dtype=[np.float32, np.int32],
                ),
                DictToListD(keys=['image', 'study_id']),
            ])
        
        super().__init__(data=df.to_dict('records'), transform=transform)


def process_scene_graph(json_path):
    """Process a single scene graph JSON file and extract relevant triplets and sentences."""
    try:
        # Attempt to parse JSON using simdjson
        scene_graph = parser.load(json_path, True)
    except:
        # Fallback to standard JSON parser
        scene_graph = json.loads(json_path.read_text())
    
    triplets = []
    for att in scene_graph['attributes']:
        for attribute in att['attributes']:
            for a in attribute:
                if 'anatomicalfinding|yes' in a or 'disease|yes' in a:
                    finding = a.split('|')[-1]
                    if finding in attrs:
                        prompt = finding + ' in the ' + list(att.keys())[0]
                        triplets.append(prompt)
    neg_label = 0
    one_hot_triplets = neg_label * np.ones(len(ALL_TRIPLETS), dtype=np.uint8)
    for p in triplets:
        if p in ALL_TRIPLETS:
            one_hot_triplets[ALL_TRIPLETS.index(p)] = 1
    
    sentences = '|'.join(list(set([a.replace('\r', '').replace('\n', '') for att in scene_graph['attributes'] if isinstance(att['phrases'], list) for a in att['phrases']])))

    # clean "IMPRESSION: " and "FINDINGS: " from the beginning of the sentences
    sentences = sentences.replace('IMPRESSION: ', '').replace('FINDINGS: ', '')
    
    if len(triplets) == 0:
        triplets = ALL_HEALTHY_TRIPLETS
    
    triplets = '|'.join(triplets)
    return [scene_graph['study_id'], sentences, triplets, one_hot_triplets]

def parallel_process(paths):
    """Parallel processing of scene graph JSON files."""
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_scene_graph, paths), total=len(paths)))
    return results

def extract_features_from_scene_graphs(chest_imagenome_path, output_path):

    assert (chest_imagenome_path / 'files').exists(), 'files folder not found in Chest Imagenome path'

    # Get all scene graph JSON file paths
    sg_path = chest_imagenome_path / 'files/chest-imagenome/1.0.0/scene_graph/scene_graph'
    print(f'Globbing all scene graphs in {sg_path} - this may take a while..')
    sg_paths = list(sg_path.rglob('*'))

    # Parallel processing of all scene graph JSON files
    results = parallel_process(sg_paths)

    # Create DataFrame from the results
    imagenome_features = pd.DataFrame(results, columns=['study_id', 'sentences', 'triplets', 'triplets_one_hot'])
    # Set index to study_id
    imagenome_features.set_index('study_id', inplace=True)
    # Drop duplicate entries (since there are multiple images per study id = one report)
    imagenome_features = imagenome_features.loc[~imagenome_features.index.duplicated(keep='first')]
    # Save to HDF5 (allows for vectors in rows
    imagenome_features.to_hdf(output_path / 'imagenome_features.h5', key='df', mode='w')

def extraxt_imagenome_attributes(chest_imagenome_path, output_path):

    df = pd.read_csv(chest_imagenome_path / 'files/chest-imagenome/1.0.0/output/attribute_relations_tabular.txt', sep='\t', encoding='utf-8')

    finding_df = df.loc[df['context']=='yes',['study_id', 'bbox', 'label_name']]
    attrs = ['lung opacity','pleural effusion','atelectasis','enlarged cardiac silhouette','pulmonary edema/hazy opacity'
        ,'pneumothorax','consolidation','fluid overload/heart failure','pneumonia']
    new_df = finding_df[['study_id', 'label_name']].drop_duplicates()
    new_df = new_df[new_df.label_name.isin(attrs)]

    attribute_df = pd.get_dummies(new_df.set_index('study_id')['label_name']).groupby(level=0).sum()
    imagenome_features = pd.read_hdf(output_path / 'imagenome_features.h5')

    attributes_df = attribute_df[attrs]
    # Create a DataFrame with zeros for the missing indices
    missing_indices = imagenome_features.index[~imagenome_features.index.isin(attribute_df.index)]
    missing_df = pd.DataFrame(0, index=missing_indices, columns=attrs)

    # Concatenate the original and the missing DataFrames
    attributes_df = pd.concat([attributes_df, missing_df])
    attributes_df.to_hdf(output_path / 'imagenome_attributes.h5', key='df', mode='w')

def main(args):

    # datestamp in the format YYMMDD
    datestamp = time.strftime('%y%m%d')
    dataset_path =  Path(args.output_path) / f'{datestamp}_imagenome_triplets_ffcv/'
    dataset_path.mkdir(parents=True, exist_ok=True)

    chest_imagenome_path = Path(args.chest_imagenome_path)

    image_size_dict = {
        'train': 336, # higher res for cropping
        'test': 224,
        'valid': 224
    }

    # Extract features from scene graphs
    print('Extracting triplets and sentences from Chest ImaGenome scene graphs from directory: ', chest_imagenome_path)
    extract_features_from_scene_graphs(chest_imagenome_path, dataset_path)

    # Extract attributes from imagenome table
    print('Extracting attributes from Chest ImaGenome scene graphs from directory: ', chest_imagenome_path)
    extraxt_imagenome_attributes(chest_imagenome_path, dataset_path)


    # Write the dataset files (convert JPGs to FFCV beton files)
    print('Writing the Chest ImaGenome dataset to ffcv beton format from directory: ', args.mimic_cxr_jpg_path)
    for split in ['train', 'test', 'valid']:
        image_size = image_size_dict[split]
        write_path = dataset_path / f'{split}.beton'
        print(f'Creatomg {split} ffcv dataset')
        dataset = ChestImaGenomeDataset(args.mimic_cxr_jpg_path, args.chest_imagenome_path, split, excluded_studies=[], image_size=image_size)
        writer = DatasetWriter(
            fname = write_path,
            fields = {
                'image': NDArrayField(np.dtype('float32'), (1, image_size, image_size)),
                'study_id': IntField(),
            }
        )
        writer.from_indexed_dataset(dataset)
        print(f'Successfully wrote {split} dataset to {write_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for the ChestImaGenome dataset.")
    parser.add_argument('--mimic_cxr_jpg_path', default='/mnt/polyaxon/data1/mimic/mimic-cxr-jpg/2.0.0', type=str)
    parser.add_argument('--chest_imagenome_path', default='/mnt/polyaxon/data1/mimic/chest-imagenome/physionet.org', type=str)
    parser.add_argument('--output_path', default='/mnt/polyaxon/data1/mimic/datasets', type=str, help='Base path for writing the output files.')
    
    args = parser.parse_args()
    main(args)