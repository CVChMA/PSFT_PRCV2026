import numpy as np
import torch
from torch.utils.data import DataLoader

from data_inputs.readCorrputionDataset import corruptionDataset
from eval import *

import sklearn.metrics as metrics

from utils.distributed_utils import get_logger
from utils.model import load_model
from utils.utils import get_args_parser, set_fixed_seed

# Initialize
args = get_args_parser()
logger, _, _ = get_logger("Test-Baseline", args)
set_fixed_seed(args.seed)

# Testing
device = torch.device("cuda")
model = load_model(args, logger).to(device)
checkpoint = torch.load(args.model_path, map_location="cpu")
if 'args' in checkpoint:
    logger.info(checkpoint['args'])
if 'epoch' in checkpoint:
    logger.info(checkpoint['epoch'])
model.load_state_dict(checkpoint['model'])
model.eval()

def test_corrupt(split):
    with torch.no_grad():
        dataset = corruptionDataset(args, split=split)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
        
        test_true = []
        test_pred = []

        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.contiguous()
            if args.model_name == 'Uni3d-B':
                colors = torch.ones_like(data).float() * 0.4
                logits = model(data, colors)
            else:
                logits = model(data)
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
