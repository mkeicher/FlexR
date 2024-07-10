import torch
from torch.utils.data import Dataset, Sampler
from typing import Union, Callable, Optional
import random
from ffcv.loader import Loader


# adapted from https://github.com/Schinkikami/PyT-ImbalancedDatasetSampler/blob/main/imbalanced_sampler/ImbalancedDatasetSampler.py
class ImbalancedDatasetSampler(Sampler):
    ''' ImbalancedDatasetSampler Class
    PyTorch Sampler class for dealing imbalanced datasets by either over- or under-sampling the classes. Can also interpolate between these modes.
    Labels can be provided via the 'labels' argument or via a callable object that is provided. The object can either return single int objects for every index in the dataset
    or a list of length len(dataset), with entries denoting the class of the datapoint at that index.
    Args:
            dataset (Dataset): The dataset the Sampler should sample from
            sampling_factor (float): Multi purpose value for letting you interpolate the size of the balanced dataset. The behaviour depends on the type of the variable. Type in [int,float].
            type(sampling_factor) == float:
                If 0.0 <= sampling_factor <= 1.0:  The size of each class will be linearly interpolated between the size of the smallest class and the largest class.
                If -1.0 < sampling_factor < 0.0: Corresponds to undersampling the largest class by a factor of -sampling_factor. Example: sampling_factor == -0.5 corresponds to undersampling the largest class by a factor of 1/2.
                If sampling_factor < 1.0: Corresponds to oversampling the smallest class by a factor of -sampling_factor. Example: sampling_factor == -2 corresponds to oversampling the smallest class by a factor of 2.
                If sampling_factor == -1.0 or sampling_factor > 1.0: Undefined behaviour. Will raise RuntimeExceptions.
            type(sampling_factor) == int:
                If sampling_factor >= 2: Set the size of each class to sampling_factor by over or undersampling
                Else: Raise Exception (to not collide with the float ranges if the wrong type is used)
            type(sampling_factor) == str:
                If sampling_factor == "oversampling": Equivalent to sampling_factor == 1.0
                If sampling_factor == "undersampling": Equivalent to sampling_factor == 0.0
            shuffle (bool) = False: If set to true every invocation of the iterator will shuffle the indices.
            labels ([list, list[list]): The labels of the dataset. Can be either a list-type object of length len(dataset), where the entry at position i, tells the class of the i-th dataset element. Alternativly, a list of lists of length num_classes is accepted, where each nested list contains the indices for the class.
            seed (int) = 0: Seed for the random number generator
            sample_negatives (bool) = True: If set to True, negative samples will be included in the dataset. Negative samples are samples where the given class is not present.
    '''

    def __init__(self, sampling_factor: Union[str, float] = None, labels: Optional[Union[list, list[list]]] = None, seed = None, sample_negatives=False, shuffle=True):

        # list of idxs of positive samples for each class
        self.num_classes = labels.shape[1]
        self.strata_indices = [np.where(labels[:,i]>0)[0] for i in range(self.num_classes)]
        self.strata_indices.append(np.where(labels.clip(0,1).sum(axis=1)==0)[0]) # add negative samples (all 0 or lower)
        
        # if enabled, add stratum with negative samples for each class
        if sample_negatives:
            self.strata_indices.extend([np.where(labels[:,i]>0)[0] for i in range(self.num_classes)])

        # remove strata with no samples
        self.strata_indices = [idxs for idxs in self.strata_indices if len(idxs) > 0]

        self.num_strata = len(self.strata_indices)
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = random.randint(0, 2**32) if seed is None else seed
        self.new_samples_every_epoch = True
        self.samples_per_epoch = ''
        samples_per_stratum = [len(i) for i in self.strata_indices]
        min_size = min(samples_per_stratum)
        max_size = max(samples_per_stratum)

        if sampling_factor is None:
            raise Exception("sampling_factor is None")
        elif sampling_factor == "oversampling":
            sampling_factor = 1.0
        elif sampling_factor == "undersampling":
            sampling_factor = 0.0
        elif sampling_factor == "median":
            sampling_factor = int(np.median(samples_per_stratum))
        elif 'shot' in sampling_factor:
            sampling_factor, self.samples_per_epoch = sampling_factor.split('shot')
            sampling_factor = int(sampling_factor)
            self.samples_per_epoch = '' if self.samples_per_epoch == '' else int(self.samples_per_epoch)
            assert sampling_factor, 'number of fewshot samples cannot be 0'
            self.new_samples_every_epoch = False
        elif 'fixed' in sampling_factor:
            sampling_factor = int(sampling_factor.split('fixed')[1])
            assert sampling_factor, 'number of samples cannot be 0'
            self.new_samples_every_epoch = True

        if type(sampling_factor) is not int:
            #Linear interpolate the class size between the largest and smallest class
            if 0.0 <= sampling_factor <= 1.0:
                inter_class_distance = abs(max_size - min_size)
                self.stratum_sample_size = min_size + int(sampling_factor * inter_class_distance)
            #Downsample the largest class by a factor
            elif -1.0 < sampling_factor < 0.0:
                self.stratum_sample_size = int(max_size * -sampling_factor)
            #Upsample the smallest class by a factor
            elif sampling_factor < -1.0:
                self.stratum_sample_size = int(min_size * -sampling_factor)
            else:
                raise NotImplementedError(
                    "Called sampling factor with value -1.0 (behaviour undefined) or a float greater than 1.0")

        else:
            #Directly set the class size
            self.stratum_sample_size = sampling_factor

        assert self.stratum_sample_size <= max_size # and self.stratum_sample_size >= min_size

        self.__indices = self.__build_indices()

    def __build_indices(self):
        # Build dataset

        indices = []
        
        generator = torch.Generator().manual_seed(self.seed + self.epoch)

        for stratum_indices in self.strata_indices:
            # Determine if we need to over- or undersample
            samples_per_stratum = len(stratum_indices)

            # Since sampling is based on positive samples classes with no positive samples are ignored
            if samples_per_stratum > 0:
                # We have fewer actual labels than required. We need to oversample
                if samples_per_stratum <= self.stratum_sample_size:
                    reps = self.stratum_sample_size // samples_per_stratum
                    sample_size = self.stratum_sample_size % samples_per_stratum
                    assert sample_size == self.stratum_sample_size - reps * samples_per_stratum
                # Otherwise we need to undersample
                else:
                    reps = 0
                    sample_size = self.stratum_sample_size

                random_samples = self.torch_sample(sample_size, len(stratum_indices), generator).numpy()
                random_indices = stratum_indices[random_samples]
                new_indices = np.concatenate([stratum_indices.repeat(reps), random_indices])

                if self.samples_per_epoch:
                    assert self.samples_per_epoch >= self.stratum_sample_size
                    new_indices = np.resize(new_indices, self.samples_per_epoch)
                indices = np.concatenate([indices, new_indices])
        
        return indices.astype('int64')

    def __iter__(self):
        if self.new_samples_every_epoch:
            self.__indices = self.__build_indices()
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            self.__indices = self.__indices[torch.randperm(len(self.__indices), generator=generator)]
        return iter(self.__indices)

    def __len__(self):
        return self.stratum_sample_size * self.num_strata

    def torch_sample(self, sample_size, range, generator):
        return torch.argsort(torch.rand(range, generator=generator))[:sample_size]
        # if len(random_ints) == 1:
        #     return random_ints
        # return random_ints
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

from typing import Sequence

import numpy as np
from torch.utils.data import DistributedSampler

from ffcv.traversal_order.base import TraversalOrder

class FfcvSamplerOrder(TraversalOrder):

    def __init__(self, 
            loader:'Loader',  
            sampler,
        ):
        super().__init__(loader)
        self.sampler = sampler
        if self.distributed:
            raise NotImplementedError('distributed sampling not implemented yet')

    def sample_order(self, epoch: int) -> Sequence[int]:
        self.sampler.set_epoch(epoch)
        indices = np.array(list(self.sampler))
        return indices

from ffcv import Loader

class CustomFfcvLoader(Loader):
    def __init__(self, sampler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.traversal_order = FfcvSamplerOrder(self, sampler)        
        self.generate_code()
        self.first_traversal_order = self.next_traversal_order()
    