import logging
import os
import sys

import torch
import torch.distributed as dist

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'  
    args.dist_url = 'env://'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def gather_all(tensor, device):
    world_size = dist.get_world_size()
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor).to(device)
    else:
        tensor = tensor.to(device)
    tensors_gather = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensors_gather, tensor)
    return torch.cat(tensors_gather, dim=0)

def make_dirs(save_dir):
    existing_versions = os.listdir(save_dir)
        
    if len(existing_versions) > 0:
        max_version = int(existing_versions[0].split("_")[-1])
        for v in existing_versions:
            ver = int(v.split("_")[-1])
            if ver > max_version:
                    max_version = ver
        version = int(max_version) + 1
    else:
        version = 0

    return f"{save_dir}/exp_{version}"

def get_logger(model_type, args):
    logger = logging.getLogger('Exp')
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sub_folder_path = os.path.join(args.save_dir, args.model_name, args.dataset)
    os.makedirs(sub_folder_path, exist_ok=True)
    sub_folder_path = make_dirs(sub_folder_path)
    os.makedirs(sub_folder_path, exist_ok=True)

    log_name = f"{model_type}-{args.model_name}-{args.train_mode}-{args.dataset}"

    if args.add_FFM:
        log_name = log_name + "-add_FFM"

    if args.add_WOLFMix:
        log_name = log_name + "-Augmentation:WOLFMix"
    elif args.add_PointWOLF:
        log_name = log_name + "-Augmentation:PointWOLF"
    elif args.add_RSMix:
        log_name = log_name + "-Augmentation:RSMix"
    else:
        log_name = log_name + "-Augmentation:None"

    pt_name = log_name + f"-train_alpha:{args.alpha}"
    log_name = log_name + f"-train_alpha:{args.alpha}.log"    
    
    file_path = os.path.join(sub_folder_path, log_name)

    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)

    logger.info(args)
   
    return logger, sub_folder_path, pt_name
