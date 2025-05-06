import os
import yaml
from pathlib import Path
import shutil
import random
import argparse

from detector.image_utils import count_mask_instances

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
args = parser.parse_args()

config_path = args.config

print(f'Config file: {config_path}')

if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config['path']['DATA_DIR'])
MODEL_DIR = Path(config['path']['MODEL_DIR'])
OUTPUT_DIR = Path(config['path']['OUTPUT_DIR'])

N_VAL_DATA = config['data']['N_VAL_DATA']

print(f'Sample {N_VAL_DATA} validation data.')

original_train_dir = DATA_DIR / 'train'
splitted_train_dir = DATA_DIR / 'train_splitted'
splitted_val_dir = DATA_DIR / 'val_splitted'

if splitted_train_dir.exists():
    shutil.rmtree(splitted_train_dir)
if splitted_val_dir.exists():
    shutil.rmtree(splitted_val_dir)

os.makedirs(splitted_train_dir)
os.makedirs(splitted_val_dir)

original_train_data = os.listdir(original_train_dir)
splitted_val_data = random.sample(original_train_data, N_VAL_DATA)
splitted_train_data = [data 
                       for data in original_train_data 
                       if data not in splitted_val_data]

train_count = [0] * 4
val_count = [0] * 4

for data in splitted_train_data:
    original_path = original_train_dir / data
    splitted_path = splitted_train_dir / data
    shutil.copytree(original_path, splitted_path)

    for class_ in range(1, 4+1):
        mask_path = splitted_path / f'class{class_}.tif'
        if mask_path.exists():
            train_count[class_-1] += count_mask_instances(mask_path)
    
for data in splitted_val_data:
    original_path = original_train_dir / data
    splitted_path = splitted_val_dir / data
    shutil.copytree(original_path, splitted_path)

    for class_ in range(1, 4+1):
        mask_path = splitted_path / f'class{class_}.tif'
        if mask_path.exists():
            val_count[class_-1] += count_mask_instances(mask_path)

train_distr_str = [f'{c}({c / sum(train_count) * 100:.1f}%)' 
                   for c in train_count]

val_distr_str = [f'{c}({c / sum(val_count) * 100:.1f}%)' 
                 for c in val_count]

print(f'Training data distribution: {train_distr_str}')
print(f'Validation data distribution: {val_distr_str}')