import os
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat

def load_data(data_folder, batch_size, train, target, num_workers=0, num_samples = None, test_ratio = None, patch_size = 9,
              src_data_file='Source.mat', tgt_data_file='Target.mat', src_label_file='Source_map.mat', 
              tgt_label_file='Target_map.mat',**kwargs):
    
    
    # 根据数据集选择不同的数据加载库
    if data_folder.split("/")[-2] == 'Houston':
        from mat73 import loadmat
    else:
        from scipy.io import loadmat 
    

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

class HSIDataset(Dataset):
    def __init__(self, data, labels, num_samples, test_ratio, patch_size):
        self.data = torch.from_numpy(data.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))
        self.patch_size = patch_size

        # Flatten data
        self.data_flattened = self.data.reshape(-1, self.data.size(-1))  # Flatten the data tensor to a 2D tensor, each row represents a pixel and its data on channels
        self.labels_flattened = self.labels.reshape(-1)  # Flatten the labels tensor to a 1D tensor, each element represents the label of a pixel

        # Exclude 0 label for stratified sampling
        exclude_indices = np.where(self.labels_flattened != 0)[0]
        stratified_data = self.data_flattened[exclude_indices]
        stratified_labels = self.labels_flattened[exclude_indices]

        # Stratified sampling
        unique_labels, counts = np.unique(stratified_labels, return_counts=True)
        class_indices = []
        for label in unique_labels:
            indices = np.where(stratified_labels == label)[0]
            class_indices.append(indices)

        sampled_indices = []
        # Sample the number of samples for each class
        for indices in class_indices:
            if num_samples is not None:
                samples_class = int(num_samples)
            else:
                samples_class = int(len(indices) * test_ratio)

            if samples_class < len(indices):
                sampled_indices.extend(np.random.choice(indices, size=samples_class, replace=False))
            else:
                sampled_indices.extend(indices)

        # 输出抽取的每个类别的样本数量
        sampled_unique_labels, sampled_counts = np.unique(stratified_labels[sampled_indices], return_counts=True)
        for label, count in zip(sampled_unique_labels, sampled_counts):
            print(f"Sampled Class {label}: {count} samples")

        # 将抽样的索引映射回原始数据
        self.sampled_indices = exclude_indices[sampled_indices]
        
        
    def __len__(self):
        return len(self.sampled_indices)  # Return the length of the flattened data tensor, which is the number of pixels
    
    def __getitem__(self, idx):
        # 获取中心像素索引
        center_idx = self.sampled_indices[idx]
        
        # 获取中心像素坐标
        center_row, center_col = np.unravel_index(center_idx, self.labels.shape)
        
        # 定义小块边长
        patch_size = self.patch_size
        
        # 获取小块数据和标签
        patch_data = []
        for row in range(center_row - patch_size // 2, center_row + patch_size // 2 + 1):
            row_data = []
            for col in range(center_col - patch_size // 2, center_col + patch_size // 2 + 1):
                row_pad = np.clip(row, 0, self.data.shape[0] - 1)
                col_pad = np.clip(col, 0, self.data.shape[1] - 1)
                row_data.append(self.data[row_pad, col_pad])
            patch_data.append(torch.stack(row_data, dim=0))  # 将每一行的像素堆叠成一维张量
        
        patch_data = torch.stack(patch_data, dim=0)  # 将小块数据堆叠成一个张量
        patch_data = patch_data.permute(2, 0, 1)  # 将小块数据的通道维度放到第一维
        patch_labels = self.labels_flattened[center_idx] # 获取中心像素的标签
        
        return patch_data, patch_labels

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
