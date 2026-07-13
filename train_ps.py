"""
@Author: Chang Ma
@Contact: szuchangma@gmail.com
"""

import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import sklearn.metrics as metrics
from data_inputs.models.Point_Selection.Point_Selection import PS
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
if args.rank == 0: 
    logger, ckpt_dir, pt_name = get_logger("Train-PS", args)
    logger.info(f"Use GPUs: {args.world_size}")
    
# Training
train_dataset = cleanDataset(args, partition='train')
val_dataset = cleanDataset(args, partition='test')
train_loader, val_loader, train_sampler = build_clean_dataloaders_ddp(train_dataset, val_dataset, args)
device = torch.device(f'cuda:{args.gpu}')
ps_model = DDP(PS(args).to(device), device_ids=[args.gpu])
cls_model = load_model(args, logger).to(device)
if args.train_mode == 'PG':
    cls_model.freeze()
if args.rank == 0: 
    for k, v in cls_model.named_parameters():
        if v.requires_grad:
            logger.info(k)
cls_model = DDP(cls_model, device_ids=[args.gpu], find_unused_parameters=True)
ps_opt, ps_scheduler = build_opti_sche_ddp(ps_model, args)
cls_opt, cls_scheduler = build_opti_sche_ddp(cls_model, args)

criterion = cal_loss
best_val_acc = 0

for epoch in range(args.epochs):
    train_sampler.set_epoch(epoch)
    start = time.time()
    loss_cls, loss_ps = ps_train_epoch(
        args, epoch, cls_model, train_loader, device, criterion,
        cls_opt, cls_scheduler,
        ps_model, ps_opt, ps_scheduler
    )
    train_loss_cls = gather_all(loss_cls.unsqueeze(0), device=device).cpu().numpy().mean()
    train_loss_ps = gather_all(loss_ps.unsqueeze(0), device=device).cpu().numpy().mean()
    val_pred, val_true = ps_val_epoch(args, cls_model, val_loader, device, ps_model)
    val_pred_all = gather_all(val_pred, device).cpu().numpy()
    val_true_all = gather_all(val_true, device).cpu().numpy()
    end = time.time()

    if args.rank == 0:
        val_acc = metrics.accuracy_score(val_true_all, val_pred_all)
       
        logger.info(
            f"Epoch {epoch} | Train Cls Loss {train_loss_cls:.4f} | "
            f"Train PS Loss {train_loss_ps:.4f} | "
            f"Val Acc {val_acc * 100:.2f}% | Epoch Time {int(end - start)}s"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            
            torch.save(
                {"args": args, "epoch": epoch + 1, "cls_model": cls_model.module.state_dict(), "ps_model": ps_model.module.state_dict()},
                os.path.join(ckpt_dir, f"best_{pt_name}.pt")
            )
            logger.info(f"Best model saved with Val Acc: {best_val_acc * 100:.2f}% at epoch {epoch + 1}")

        if (epoch + 1) % 50 == 0:
            torch.save(
                {"args": args, "epoch": epoch + 1, "cls_model": cls_model.module.state_dict(), "ps_model": ps_model.module.state_dict()},
                os.path.join(ckpt_dir, f"{pt_name}_epoch_{epoch + 1}.pt")
            )
            logger.info(f"Checkpoint model saved at epoch {epoch + 1} with Val Acc: {val_acc * 100:.2f}%")    
