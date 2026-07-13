import torch
from augmentation.RSMix import rsmix_provider
from data_inputs.models.Uni3d_B.Uni3d_B import Uni3d_B
from utils.pointSelection import *
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
import yaml
from types import SimpleNamespace

NUM_POINTS = 1024

def load_yaml(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def load_model(args, logger):
    point_encoder = None
    state_dict = None
    if args.model_name == 'ULIP-2':
        config = load_yaml('data_inputs/models/Point_BERT/Point_BERT.yaml')
        if args.train_mode == 'FT':
            from data_inputs.models.Point_BERT.Point_BERT import PointTransformer
            point_encoder = PointTransformer(config, args)
        elif args.train_mode == 'PG':
            from data_inputs.models.Point_BERT.Point_BERT_PG import PointTransformer
            point_encoder = PointTransformer(config, args)
        checkpoint = torch.load('data_inputs/pretrained/pretrained_models_ckpt_zero-shot_classification_pointbert_ULIP-2.pt', map_location='cpu')
        state_dict = checkpoint['state_dict']
        st = {}
        for k, v in state_dict.items():
            if 'module.point_encoder.' in k:
                st[k.replace('module.point_encoder.', '')] = v
        missing_keys, unexpected_keys = point_encoder.load_state_dict(st, False) 
        if logger is not None:
            logger.info(f"missing_keys:{missing_keys}")
            logger.info(f"unexpected_keys:{unexpected_keys}")
    elif args.model_name == 'Point-BERT':
        config = load_yaml('data_inputs/models/Point_BERT/Point_BERT.yaml')
        if args.train_mode == 'FT':
            from data_inputs.models.Point_BERT.Point_BERT import PointTransformer
            point_encoder = PointTransformer(config, args)
        elif args.train_mode == 'PG':
            from data_inputs.models.Point_BERT.Point_BERT_PG import PointTransformer
            point_encoder = PointTransformer(config, args)
        checkpoint = torch.load('data_inputs/pretrained/Point-BERT.pth', map_location='cpu')
        state_dict = checkpoint['base_model']
        st = {}
        for k, v in state_dict.items():
            if 'transformer_q' in k:
                st[k.replace('transformer_q.', '')] = v
        missing_keys, unexpected_keys = point_encoder.load_state_dict(st, False) 
        if logger is not None:
            logger.info(f"missing_keys:{missing_keys}")
            logger.info(f"unexpected_keys:{unexpected_keys}")
    elif args.model_name == 'Point-MAE':
        config = load_yaml('data_inputs/models/Point_MAE/Point_MAE.yaml')
        if args.train_mode == 'FT':
            from data_inputs.models.Point_MAE.Point_MAE import PointTransformer
            point_encoder = PointTransformer(config, args)
        elif args.train_mode == 'PG':
            from data_inputs.models.Point_MAE.Point_MAE_PG import PointTransformer
            point_encoder = PointTransformer(config, args)
        checkpoint = torch.load('data_inputs/pretrained/Point-MAE.pth', map_location='cpu')
        state_dict = checkpoint['base_model']
        st = {}
        for k, v in state_dict.items():
            if 'module.MAE_encoder.' in k:
                st[k.replace('module.MAE_encoder.', '')] = v
        missing_keys, unexpected_keys = point_encoder.load_state_dict(st, False) 
        if logger is not None:
            logger.info(f"missing_keys:{missing_keys}")
            logger.info(f"unexpected_keys:{unexpected_keys}")
    elif args.model_name == 'Uni3d-B':
        config = load_yaml('data_inputs/models/Uni3d_B/Uni3d_B.yaml')
        point_encoder = Uni3d_B(config, args)
        checkpoint = torch.load('data_inputs/pretrained/Pretrianed_Uni3d_B_Ensembled.pt', map_location='cpu')
        state_dict = checkpoint['module']
        st = {}
        for k, v in state_dict.items():
            if 'point_encoder.' in k:
                st[k.replace('point_encoder.', '')] = v
        missing_keys, unexpected_keys = point_encoder.load_state_dict(st, False) 
        if logger is not None:
            logger.info(f"missing_keys:{missing_keys}")
            logger.info(f"unexpected_keys:{unexpected_keys}")
  
    return point_encoder


def ps_train_epoch(args, epoch, cls_model, loader, device, criterion,
                cls_optimizer, cls_scheduler,
                ps_model, ps_optimizer, ps_scheduler):

    cls_model.train()
    ps_model.train()

    for data, label in loader:
        r = np.random.rand(1)
        if args.add_WOLFMix and r < args.rsmix_prob:
            data, lam, label, label_b = rsmix_provider.rsmix(data.cpu().numpy(), label.cpu().numpy(), beta=args.beta, n_sample=args.nsample, KNN=args.knn)
            lam = torch.FloatTensor(lam)
            label_b = torch.LongTensor(label_b)
            data = torch.FloatTensor(data)
            label = torch.LongTensor(label) 
            lam, label_b = lam.to(device), label_b.to(device).squeeze()

        data, label = data.to(device), label.to(device)
        data = data.permute(0, 2, 1).contiguous()      
        num_rand_points = torch.randint(256, data.shape[-1], (1,)).item()
        ppc_critical, _ = extract_discrete_critical(data, ps_model, num_rand_points)  
        if args.model_name == 'Uni3d-B':
            colors = torch.ones_like(ppc_critical).float() * 0.4
            logits_cls = cls_model(ppc_critical.transpose(1, 2).contiguous(), colors)
        else:
            logits_cls = cls_model(ppc_critical.transpose(1, 2).contiguous()) 
    
        logits_ps, _ = ps_model(ppc_critical)
        if args.add_WOLFMix and r < args.rsmix_prob:
            loss_cls = 0
            loss_ps = 0
            for i in range(args.train_batch_size):
                loss_tmp = criterion(logits_cls[i].unsqueeze(0), label[i].unsqueeze(0).long())*(1-lam[i]) \
                    + criterion(logits_cls[i].unsqueeze(0), label_b[i].unsqueeze(0).long())*lam[i]
                loss_cls += loss_tmp
                loss_tmp = criterion(logits_ps[i].unsqueeze(0), label[i].unsqueeze(0).long())*(1-lam[i]) \
                    + criterion(logits_ps[i].unsqueeze(0), label_b[i].unsqueeze(0).long())*lam[i]
                loss_ps += loss_tmp
            loss_cls = loss_cls/args.train_batch_size
            loss_ps = loss_ps/args.train_batch_size
        else:
            loss_cls = criterion(logits_cls, label)
            loss_ps = criterion(logits_ps, label)

        loss_cls.backward()
        loss_ps.backward()

        ps_optimizer.step()
        ps_scheduler.step(epoch)
        cls_optimizer.step()
        cls_scheduler.step(epoch)

        cls_optimizer.zero_grad()
        ps_optimizer.zero_grad()

    return loss_cls, loss_ps

@torch.no_grad()
def ps_val_epoch(args, cls_model, loader, device, ps_model):
    cls_model.eval()
    ps_model.eval()

    pred_all, labels_all = [], []

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        data = data.permute(0, 2, 1).contiguous()
        num_rand_points = torch.randint(256, data.shape[-1], (1,)).item()
        ppc_critical, _ = extract_discrete_critical(data, ps_model, num_rand_points)
        if args.model_name == 'Uni3d-B':
            colors = torch.ones_like(ppc_critical).float() * 0.4
            logits = cls_model(ppc_critical.transpose(1, 2).contiguous(), colors)
        else:
            logits = cls_model(ppc_critical.transpose(1, 2).contiguous())  

        pred_all.append(logits.max(dim=1)[1].detach().cpu().numpy())
        labels_all.append(label.cpu().numpy())

    return np.concatenate(pred_all), np.concatenate(labels_all)


def baseline_train_epoch(args, epoch, model, loader, device, criterion, optimizer, scheduler):
    model.train()
    for data, label in loader:
        data, label = data.to(device), label.to(device)
        if args.model_name == 'Uni3d-B':
            colors = torch.ones_like(data).float() * 0.4
            logits = model(data, colors)
        else:
            logits = model(data)    
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch)
        optimizer.zero_grad()
        
    return loss

@torch.no_grad()
def baseline_val_epoch(args, model, loader, device):
    model.eval()
    pred_all, labels_all = [], []

    for data, label in loader:
        data, label = data.to(device), label.to(device)
        if args.model_name == 'Uni3d-B':
            colors = torch.ones_like(data).float() * 0.4
            logits = model(data, colors)
        else:
            logits = model(data)   
        pred_all.append(logits.max(dim=1)[1].detach().cpu().numpy())
        labels_all.append(label.cpu().numpy())

    return np.concatenate(pred_all), np.concatenate(labels_all)

def build_opti_sche_ddp(base_model, args):
    
    def add_weight_decay(model, weight_decay=1e-5):
        decay = []
        no_decay = []
        for name, param in model.module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name :
                # print(name)
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    param_groups = add_weight_decay(base_model, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr)

    scheduler = CosineLRScheduler(optimizer,
            t_initial=args.epochs,
            lr_min=1e-6,
            warmup_lr_init=1e-6,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True)
    
    return optimizer, scheduler

def build_opti_sche(base_model, args):
    
    def add_weight_decay(model, weight_decay=1e-5):
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name :
                # print(name)
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]
    param_groups = add_weight_decay(base_model, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr)

    scheduler = CosineLRScheduler(optimizer,
            t_initial=args.epochs,
            lr_min=1e-6,
            warmup_lr_init=1e-6,
            warmup_t=args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True)
    
    return optimizer, scheduler
