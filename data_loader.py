import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

def load_data(data_folder, batch_size, train, target, num_workers=0, num_samples = None, test_ratio = None, patch_size = 9,
              src_data_file='Source.mat', tgt_data_file='Target.mat', src_label_file='Source_map.mat', 
              tgt_label_file='Target_map.mat',**kwargs):
    

    if not target:
        data_file = os.path.join(data_folder, src_data_file)
        label_file = os.path.join(data_folder, src_label_file)
    else:
        data_file = os.path.join(data_folder, tgt_data_file)
        label_file = os.path.join(data_folder, tgt_label_file)

    data = loadmat(data_file)
    data = data['ori_data']  # shape=(height, width,num_bands)

    labels = loadmat(label_file)
    labels = labels['map'].squeeze()  # shape=(num_samples,)

    dataset = HSIDataset(data, labels, num_samples, test_ratio, patch_size)

    data_loader = get_data_loader(dataset, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    
    n_class = len(np.unique(labels))

    return data_loader, n_class

def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0
