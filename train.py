import torch
from torch.utils.data import DataLoader

import argparse
import yaml
import os
from pathlib import Path

from detector import Detector, Trainer
from detector.dataset import TrainDataset, aggregate_batch
from detector.transform import train_transform, val_transform

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
args = parser.parse_args()

config_path = args.config
if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

print(f'Use config file "{config_path}"')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config['path']['DATA_DIR'])
MODEL_DIR = Path(config['path']['MODEL_DIR'])
OUTPUT_DIR = Path(config['path']['OUTPUT_DIR'])

BATCH_SIZE = config['train']['BATCH_SIZE']
LEARNING_RATE = config['train']['LEARNING_RATE']
MAX_EPOCHES = config['train']['MAX_EPOCHES']
EARLY_STOP = config['train']['EARLY_STOP']

train_dataset = TrainDataset(DATA_DIR / 'train_splitted', train_transform)
val_dataset = TrainDataset(DATA_DIR / 'val_splitted', val_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=aggregate_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=aggregate_batch)

model = Detector()
n_parameters = sum(p.numel() for p in model.parameters())

print(f'Model name: {model.model_name}')
print(f'Number of params: {n_parameters:,}')
    
assert n_parameters <= 200 * 1024 * 1024

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
loss_weights = {
    'loss_classifier': 1.0,
    'loss_box_reg': 1.0,
    'loss_objectness': 1.0,
    'loss_rpn_box_reg': 1.0,
    'loss_mask': 1.0,
}

trainer = Trainer()
trainer.train(
    model,
    train_loader,
    val_loader,
    MODEL_DIR,
    MAX_EPOCHES,
    optimizer,
    scheduler,
    loss_weights,
    EARLY_STOP,
)