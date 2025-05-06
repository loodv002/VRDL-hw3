import os
from pathlib import Path
import torch
import numpy as np
import tifffile
import random

from .image_utils import read_mask

augment_tranforms = [
    lambda x: x,
    lambda x: x.flip(-1),
    lambda x: x.flip(-2),
    lambda x: x.transpose(-2, -1).flip(-1),
    lambda x: x.transpose(-2, -1).flip(-2),
]

def aggregate_batch(data):
    images = []
    targets = []

    for image, target in data:
        images.append(image)
        targets.append(target)
    
    return images, targets

class TrainDataset:
    def __init__(self, root_dir, transform):
        self.root = Path(root_dir)
        self.data = os.listdir(root_dir)
        self.transform = transform

        self.BBOX_PADDING = 10

    def _get_image(self, image_dir: Path, augment_transform):
        image = tifffile.imread(image_dir / 'image.tif')[:, :, :3]
        image = self.transform(image)
        image = augment_transform(image)
        return image
    
    def _get_target(self, image_dir: Path, augment_transform):
        boxes = []
        labels = []
        masks = []

        for class_id in range(1, 4 + 1):
            mask_path = image_dir / f'class{class_id}.tif'
            if not mask_path.exists(): continue

            mask = torch.Tensor(read_mask(mask_path))
            mask = augment_transform(mask)
            sub_boxes, sub_masks = self._to_boxes_masks(mask)
            n_instances = len(sub_boxes)

            boxes.extend(sub_boxes)
            labels.extend([class_id] * n_instances)
            masks.extend(sub_masks)

        # Required to suprees warnings
        masks = np.array(masks)

        return {
            'boxes': torch.FloatTensor(boxes),
            'labels': torch.LongTensor(labels),
            'masks': torch.IntTensor(masks),
        }

    def _to_boxes_masks(self, mask: torch.Tensor):
        h, w = mask.shape
        instances = np.unique(mask)
        boxes = []
        masks = []
        
        for instance in instances:
            if instance == 0: continue

            instance_mask = (mask == instance)
            
            ys, xs = np.where(instance_mask)
            x1 = max(xs.min() - self.BBOX_PADDING, 0)
            y1 = max(ys.min() - self.BBOX_PADDING, 0)
            x2 = min(xs.max() + self.BBOX_PADDING, w - 1)
            y2 = min(ys.max() + self.BBOX_PADDING, h - 1)

            boxes.append([x1, y1, x2, y2])
            masks.append(instance_mask)

        return boxes, masks
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        image_dir = self.root / self.data[i]
        augment_tranform = random.choice(augment_tranforms)
        
        image = self._get_image(image_dir, augment_tranform)
        target = self._get_target(image_dir, augment_tranform)

        return image, target
        
class TestDataset:
    def __init__(self, root_dir, transform):
        self.root = Path(root_dir)
        self.data = os.listdir(root_dir)
        self.transform = transform

    def _get_image(self, image_path: Path):
        image = tifffile.imread(image_path)[:, :, :3]
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_dir = self.root / self.data[i]
        return self.data[i], self._get_image(image_dir)