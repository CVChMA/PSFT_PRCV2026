"""
@Author: Chang Ma
@Contact: szuchangma@gmail.com
"""

import torch
import argparse
import random
import os
import numpy as np

def set_fixed_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def cal_model_parm_nums(model):
    total = 0
    for param in model.parameters():
        if param.requires_grad:  
            total += param.numel()

    if total >= 1e9:
        return "{:.2f}B".format(total / 1e9)
    elif total >= 1e6:
        return "{:.2f}M".format(total / 1e6)
    elif total >= 1e3:
        return "{:.2f}K".format(total / 1e3)
    else:
        return str(total)

def get_args_parser():
    parser = argparse.ArgumentParser(description='PSFT',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## seed
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    ## checkpoint
    parser.add_argument('--save_dir', type=str, default='./data_outputs', help='Directory to save logs and checkpoints')
    parser.add_argument('--model_path', type=str, default='', help='Path to the trained model')

    ## batch size
    parser.add_argument('--train_batch_size', default=16, type=int, help='Training batch size')
    parser.add_argument('--test_batch_size', default=1, type=int, help='Testing batch size')

    ## optimization
    parser.add_argument('--epochs', default=300, type=int, help='Total number of training epochs ')
    parser.add_argument('--lr', default=0.0005, type=float, help='Learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for optimizer')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of warm-up epochs for learning rate scheduler')

    ## model
    parser.add_argument(
    '--model_name',
    type=str,
    default='ULIP-2',
    choices=['Point-MAE', 'Point-BERT', 'ULIP-2', 'Uni3d-B'],
    help='Name of the model'
    )
    parser.add_argument(
    '--train_mode',
    type=str,
    default='FT',
    choices=['FT', 'PG'],
    help='Training mode'
    )

    ## hyperparameter
    parser.add_argument('--alpha', type=float, default=0.5, help='Alpha parameter of the Beta distribution')
    parser.add_argument('--test_lam', type=float, default=0.618, help='Test Lam parameter of the Beta distribution')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings of PS')

    ## Ablation
    parser.add_argument('--add_FFM', action='store_true', help='Whether to add FFM module')

    # Augmentation settings
    parser.add_argument('--add_PointWOLF', action='store_true', help='Use PointWOLF')
    parser.add_argument('--add_RSMix', action='store_true', help='Use RSMix')
    parser.add_argument('--add_WOLFMix', action='store_true', help='Use WOLFMix')

    # PointWOLF
    parser.add_argument('--w_num_anchor', type=int, default=4, help='Num of anchor point' ) 
    parser.add_argument('--w_sample_type', type=str, default='fps', help='Sampling method for anchor point, option : (fps, random)') 
    parser.add_argument('--w_sigma', type=float, default=0.5, help='Kernel bandwidth')  

    parser.add_argument('--w_R_range', type=float, default=10, help='Maximum rotation range of local transformation')
    parser.add_argument('--w_S_range', type=float, default=3, help='Maximum scailing range of local transformation')
    parser.add_argument('--w_T_range', type=float, default=0.25, help='Maximum translation range of local transformation')

    # RSMix
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=1.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512, help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')

    ## dataset setting: ModelNet-C, Objaverse-C, ScanObjectNN-C and ModelNet40-C
    subparsers = parser.add_subparsers(title="dataset setting", dest="dataset")

    # --- ModelNet-C --- #
    ModelNetC = subparsers.add_parser("ModelNet-C",
                                  description='Dataset parser for training on ModelNet-C',
                                  add_help=True,
                                  formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  help="Dataset parser for training on ModelNet-C")

    ModelNetC.add_argument("--train_dir", type=str, default='./data_inputs/data/modelnet40_ply_hdf5_2048', help="ModelNet-C train directory")
    ModelNetC.add_argument("--test_dir", type=str, default='./data_inputs/data/modelnet_c', help="ModelNet-C test directory")
    ModelNetC.add_argument("--nb_cls", type=int, default=40, help="number of classes in ModelNet-C")

    # --- Objaverse-C --- #
    ObjaverseC = subparsers.add_parser("Objaverse-C",
                                   description='Dataset parser for training on Objaverse-C',
                                   add_help=True,
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                   help="Dataset parser for training on Objaverse-C")

    ObjaverseC.add_argument("--train_dir", type=str, default='./data_inputs/data/objaverse_lvis_mc', help="Objaverse-C train directory")
    ObjaverseC.add_argument("--test_dir", type=str, default='./data_inputs/data/objaverse_c', help="Objaverse-C test directory")
    ObjaverseC.add_argument("--nb_cls", type=int, default=1156, help="Number of classes in Objaverse-C")


    # --- ScanObjectNN-C --- #
    ScanObjectNNC = subparsers.add_parser("ScanObjectNN-C",
                                        description='Dataset parser for training on ScanObjectNN-C',
                                        add_help=True,
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        help="Dataset parser for training on ScanObjectNN-C")

    ScanObjectNNC.add_argument("--train_dir", type=str, default='./data_inputs/data/ScanObjectNN/h5_files/main_split', help="ScanObjectNN-C train directory")
    ScanObjectNNC.add_argument("--test_dir", type=str, default='./data_inputs/data/scanobjectnn_c', help="ScanObjectNN-C test directory")
    ScanObjectNNC.add_argument("--nb_cls", type=int, default=15, help="Number of classes in ScanObjectNN-C")


    # --- ModelNet40-C --- #
    ModelNet40C = subparsers.add_parser("ModelNet40-C",
                                    description='Dataset parser for training on ModelNet40-C',
                                    add_help=True,
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    help="Dataset parser for training on ModelNet40-C")

    ModelNet40C.add_argument("--train_dir", type=str, default='./data_inputs/data/modelnet40_ply_hdf5_2048', help="ModelNet40-C train directory")
    ModelNet40C.add_argument("--test_dir", type=str, default='./data_inputs/data/modelnet40_c', help="ModelNet40-C test directory")
    ModelNet40C.add_argument("--nb_cls", type=int, default=40, help="Number of classes in ModelNet40-C")


    return parser.parse_args()
