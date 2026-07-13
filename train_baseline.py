"""
@Author: Chang Ma
@Contact: szuchangma@gmail.com
"""

import os
import time
import torch
import sklearn.metrics as metrics
from utils.utils import *
from utils.distributed_utils import *
from utils.model import *
from utils.loss import cal_loss
from data_inputs.readCleanDataset import build_clean_dataloaders, cleanDataset

# Initialize
args = get_args_parser()
set_fixed_seed(args.seed)
logger, ckpt_dir, pt_name = get_logger("Train-Baseline", args)

# Training
train_dataset = cleanDataset(args, partition='train')
val_dataset = cleanDataset(args, partition='test')
train_loader, val_loader = build_clean_dataloaders(train_dataset, val_dataset, args)
device = torch.device('cuda')
model = load_model(args, logger).to(device)
opt, scheduler= build_opti_sche(model, args)
criterion = cal_loss
best_val_acc = 0

for epoch in range(args.epochs):
    start = time.time()
    loss = baseline_train_epoch(
        args, epoch, model, train_loader, device, criterion, opt, scheduler
    )
    train_loss = loss.item()
    val_pred, val_true = baseline_val_epoch(args, model, val_loader, device)
    end = time.time()


    val_acc = metrics.accuracy_score(val_true, val_pred)
    logger.info(
        f"Epoch {epoch + 1} | Train Loss {train_loss:.4f} | "
        f"Val Acc {val_acc * 100:.2f}% | Epoch Time {int(end - start)}s"
    )
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        torch.save(
            {"model": model.state_dict()},
            os.path.join(ckpt_dir, f"{pt_name}.pt")
        )
        logger.info(f"save!! best val: {best_val_acc * 100:.2f}%")       
