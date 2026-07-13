"""
@Author: Chang Ma
@Contact: szuchangma@gmail.com
"""

import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import sklearn.metrics as metrics
from utils.utils import *
from utils.distributed_utils import *
from utils.model import *
from utils.loss import cal_loss
from data_inputs.readCleanDataset import build_clean_dataloaders_ddp, cleanDataset

# Initialize
args = get_args_parser()
init_distributed_mode(args)
set_fixed_seed(args.seed)
logger = None
ckpt_dir = None
pt_name = None
if args.rank == 0: 
    logger, ckpt_dir, pt_name = get_logger("Train-Baseline", args)
    logger.info(f"Use GPUs: {args.world_size}") 

# Training
train_dataset = cleanDataset(args, partition='train')
val_dataset = cleanDataset(args, partition='test')
train_loader, val_loader, train_sampler = build_clean_dataloaders_ddp(train_dataset, val_dataset, args)
device = torch.device(f'cuda:{args.gpu}')
model = load_model(args, logger).to(device)
if args.train_mode == 'PG':
    model.freeze()
if args.rank == 0: 
    for k, v in model.named_parameters():
        if v.requires_grad:
            logger.info(k)
model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
opt, scheduler= build_opti_sche_ddp(model, args)
criterion = cal_loss
best_val_acc = 0

for epoch in range(args.epochs):
    train_sampler.set_epoch(epoch)
    start = time.time()
    loss = baseline_train_epoch(
        args, epoch, model, train_loader, device, criterion, opt, scheduler
    )
    train_loss = gather_all(loss.unsqueeze(0), device=device).cpu().numpy().mean()
    val_pred, val_true = baseline_val_epoch(args, model, val_loader, device)
    val_pred_all = gather_all(val_pred, device).cpu().numpy()
    val_true_all = gather_all(val_true, device).cpu().numpy()
    end = time.time()

    if args.rank == 0:
        val_acc = metrics.accuracy_score(val_true_all, val_pred_all)
        logger.info(
            f"Epoch {epoch + 1} | Train Loss {train_loss:.4f} | "
            f"Val Acc {val_acc * 100:.2f}% | Epoch Time {int(end - start)}s"
        )
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            
            torch.save(
                {"args": args, "epoch": epoch + 1, "model": model.module.state_dict()},
                os.path.join(ckpt_dir, f"best_{pt_name}.pt")
            )
            logger.info(f"Best model saved with Val Acc: {best_val_acc * 100:.2f}% at epoch {epoch + 1}")

        if (epoch + 1) % 50 == 0:
            torch.save(
                {"args": args, "epoch": epoch + 1, "model": model.module.state_dict()},
                os.path.join(ckpt_dir, f"{pt_name}_epoch_{epoch + 1}.pt")
            )
            logger.info(f"Checkpoint model saved at epoch {epoch + 1} with Val Acc: {val_acc * 100:.2f}%")       
