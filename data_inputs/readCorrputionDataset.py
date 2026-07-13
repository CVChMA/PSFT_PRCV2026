import os
import numpy as np
from torch.utils.data import Dataset

from data_inputs.readCleanDataset import load_h5

class corruptionDataset(Dataset):
    def __init__(self, args, split):
        if args.dataset in ['ModelNet-C', 'ScanObjectNN-C']:
            h5_path = os.path.join(args.test_dir, split + '.h5')
            self.data, self.label = load_h5(h5_path)
        elif args.dataset == 'Objaverse-C':
            h5_path = os.path.join(args.test_dir, 'objaverse_' + split + '.h5')
            self.data, self.label = load_h5(h5_path)
        elif args.dataset == 'ModelNet40-C':
            data_path = os.path.join(args.test_dir, 'data_' + split + '.npy')
            label_path = os.path.join(args.test_dir, 'label.npy')
            self.data = np.load(data_path).astype('float32')
            self.label = np.load(label_path).astype('int64')
          
    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]

        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]    
