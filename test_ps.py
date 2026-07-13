"""
@Author: Chang Ma
@Contact: szuchangma@gmail.com
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_inputs.models.Point_Selection.Point_Selection import PS
from data_inputs.readCorrputionDataset import corruptionDataset
from eval import *

import sklearn.metrics as metrics

from utils.distributed_utils import get_logger
from utils.model import load_model
from utils.pointSelection import extract_discrete_critical
from utils.utils import get_args_parser, set_fixed_seed

# Initialize
args = get_args_parser()
logger, _, _  = get_logger('Test-PS', args)
set_fixed_seed(args.seed)

# Testing
device = torch.device("cuda")
cls_model = load_model(args, logger).to(device)
ps_model = PS(args).to(device)
model_path = args.model_path
checkpoint = torch.load(model_path, map_location="cpu")
if 'args' in checkpoint:
    logger.info(checkpoint['args'])
if 'epoch' in checkpoint:
    logger.info(checkpoint['epoch'])
st1_cls = checkpoint['cls_model']
st2_ps = checkpoint['ps_model']
cls_model.load_state_dict(st1_cls)
ps_model.load_state_dict(st2_ps)

cls_model.eval()
ps_model.eval()

def test_corrupt(split):
    with torch.no_grad():
        dataset = corruptionDataset(args, split=split)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        
        test_true = []
        test_pred = []

        for m_iter, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device).squeeze()
            
            ppc_critical, _ = extract_discrete_critical(data.transpose(1, 2).contiguous(), ps_model)
            if args.model_name == 'Uni3d-B':
                colors = torch.ones_like(ppc_critical).float() * 0.4
                logits = cls_model(ppc_critical.transpose(1, 2).contiguous(), colors.transpose(1, 2).contiguous())
            else:
                logits = cls_model(ppc_critical.transpose(1, 2).contiguous())

            pred = logits.max(dim=1)[1]
            test_true.append(label.unsqueeze(0).cpu().numpy())
            test_pred.append(pred.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        test_acc = metrics.accuracy_score(test_true, test_pred)

        return {'acc': test_acc}

if args.dataset == "ModelNet40-C":
    evalModelNet40C(logger, test_corrupt)
else:
    evalCorruptDataset(logger, test_corrupt, args)
