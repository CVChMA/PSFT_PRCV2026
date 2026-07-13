"""
@Author: Chang Ma
@Contact: szuchangma@gmail.com
"""

import os

import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from augmentation.PointWOLF.PointWOLF import PointWOLF

NUM_POINTS = 1024

class cleanDataset(Dataset):

    def __init__(self, args, partition='train'):
        self.partition = partition
        self.PointWOLF = PointWOLF(args) if (args.add_WOLFMix or args.add_PointWOLF) else None
        if args.dataset in ['ModelNet-C', 'ModelNet40-C']:
            self.data, self.label = load_ModelNet40(args, partition)
        elif args.dataset == 'Objaverse-C':
            self.data, self.label = load_Objaverse(args, partition)
        elif args.dataset == 'ScanObjectNN-C':
            self.data, self.label = load_ScanObjectNN(args, partition)
            
    def __getitem__(self, item):
        pointCloud = self.data[item][:NUM_POINTS]
        label = self.label[item]

        if self.partition == 'train':
            np.random.shuffle(pointCloud)
            if self.PointWOLF is not None:
                origin, pointCloud = self.PointWOLF(pointCloud)
            pointCloud = translate_pointCloud(pointCloud)

        return pointCloud, label

    def __len__(self):
        return self.data.shape[0]
      
def load_ModelNet40(args, partition):
    all_data = []
    all_label = []
    for h5_name in glob.glob(
            os.path.join(args.train_dir, 'ply_data_%s*.h5' % partition)):
        data, label = load_h5(h5_name)
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

# def load_Objaverse(args, partition):
#     filename = os.path.join(args.train_dir, f"{partition}_list_lvis.txt")
#     data_list=[]
#     label_list=[]

#     with open(filename) as f:
#         lines = f.readlines()
#         for line in lines:
#             splits = line.strip().split(',')
#             data_path = f"{args.train_dir}{splits[-1]}"
#             label = np.array(splits[0], dtype=np.int64)
#             data_dict = np.load(data_path,allow_pickle=True).item()
#             data = data_dict['xyz'].astype(np.float32)
#             data = pc_normalize(data)
#             # print(data.shape)
#             data_list.append(np.expand_dims(data, axis=0))
#             label_list.append(np.expand_dims(label, axis=0))

#     all_data = np.concatenate(data_list, axis=0)
#     all_label = np.concatenate(label_list, axis=0)
#     return all_data, all_label

def load_Objaverse(args, partition):
    if partition == 'train':
        dict = np.load(os.path.join(args.train_dir, f'train.npz'),allow_pickle=True)
        data = dict["data"]   
        label = dict["label"]
    else:
        dict = np.load(os.path.join(args.train_dir, f'test.npz'),allow_pickle=True)
        data = dict["data"]   
        label = dict["label"]
    return data, label

def load_ScanObjectNN(args, partition):
    if partition == 'train':
        h5_path = os.path.join(args.train_dir, f'{partition}ing_objectdataset_augmentedrot_scale75.h5')  
    else:
        h5_path = os.path.join(args.train_dir, f'{partition}_objectdataset_augmentedrot_scale75.h5')
    data, label = load_h5(h5_path)
    return data, label

def load_h5(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label
    
def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

def translate_pointCloud(pointCloud):
    _, C = pointCloud.shape
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[C])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[C])
    translated_pointCloud = np.add(np.multiply(pointCloud, xyz1), xyz2).astype('float32')
    return translated_pointCloud

def build_clean_dataloaders_ddp(train_dataset, val_dataset, args):

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, args.train_batch_size, drop_last=True)
    nw = min([os.cpu_count(), args.train_batch_size if args.train_batch_size > 1 else 0, 8])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        pin_memory=True,
        num_workers=nw
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.train_batch_size,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=nw
    )

    return train_loader, val_loader, train_sampler

def build_clean_dataloaders(train_dataset, val_dataset, args):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=min([os.cpu_count(), 8])
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=min([os.cpu_count(), 8])
    )

    return train_loader, val_loader
