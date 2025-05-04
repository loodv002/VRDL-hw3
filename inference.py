import torch
from torch.utils.data import DataLoader

import argparse
import os
import yaml
from tqdm import tqdm
import json
import zipfile
from pathlib import Path

from detector import Detector
from detector.transform import val_transform
from detector.dataset import TestDataset
from detector.format_utils import outputs_to_annotations

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config.yml', help='config file path')
parser.add_argument('--checkpoint', required=True, help='checkpoint name')
args = parser.parse_args()

config_path = args.config
checkpoint = args.checkpoint

print(f'Config file: {config_path}')
print(f'Checkpoint: {checkpoint}')

if not os.path.exists(config_path):
    print(f'Config file "{config_path}" not exist.')
    exit()

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

DATA_DIR = Path(config['path']['DATA_DIR'])
MODEL_DIR = Path(config['path']['MODEL_DIR'])
OUTPUT_DIR = Path(config['path']['OUTPUT_DIR'])

BATCH_SIZE = config['train']['BATCH_SIZE']

test_dataset = TestDataset(DATA_DIR / 'test_release', val_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda batch: batch)

filename_to_id = {}
with open(DATA_DIR / 'test_image_name_to_ids.json', 'r') as f:
    for mapping in json.load(f):
        filename = mapping['file_name']
        image_id = mapping['id']
        filename_to_id[filename] = image_id

device = torch.device('cuda' 
                      if torch.cuda.is_available()
                      else 'cpu')

model = Detector()
model.load_state_dict(torch.load(f'{MODEL_DIR}/{checkpoint}.pth'))
model.to(device)
model.eval()

annotations = []
with torch.no_grad():
    for data in tqdm(test_loader, ncols=100):
        filenames = [filename for filename, _ in data]
        images = [image.to(device) for _, image in data]

        outputs = model(images)
        image_ids = [filename_to_id[filename]
                     for filename in filenames]
        
        annotations.extend(
            outputs_to_annotations(outputs, image_ids)
        )

with open(OUTPUT_DIR / f'{checkpoint}.json', 'w') as f:
    json.dump(annotations, f, indent=4)

output_zip_path = OUTPUT_DIR / f'{checkpoint}.zip'
output_json_path = OUTPUT_DIR / f'{checkpoint}.json'

with zipfile.ZipFile(output_zip_path, mode='w') as f:
    f.write(output_json_path, 'test-results.json')