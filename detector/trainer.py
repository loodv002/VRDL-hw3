import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from functools import partial

from typing import Dict, Tuple, Optional, Any

from .model import Detector
from .cuda_utils import clear_cache_on_leave, suppress_OOM
from .metrics_utils import coco_ap50
from .format_utils import *


class Trainer:
    def __init__(self):
        self.device = torch.device('cuda' 
                                   if torch.cuda.is_available()
                                   else 'cpu')
        
    @staticmethod
    def weighted_loss(loss_dict: Dict[str, torch.Tensor], loss_weights: Dict[str, float]):
        return sum(
            value * loss_weights.get(key, 1.0)
            for key, value in loss_dict.items()
        )
    
    @clear_cache_on_leave
    @suppress_OOM()
    def _train_batch(self, model, optimizer, loss_function, images, targets):
        model.train()
        
        images = [image.to(self.device) for image in images]
        targets = [
            {
                key: value.to(self.device)
                for key, value in target.items()
            }
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = loss_function(loss_dict)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.detach().cpu().item()

    @clear_cache_on_leave
    @suppress_OOM()
    def _val_batch(self, model, loss_function, images, targets):
        model.train()
        
        images = [image.to(self.device) for image in images]
        targets = [
            {
                key: value.to(self.device)
                for key, value in target.items()
            }
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = loss_function(loss_dict).detach().cpu().item()

        model.eval()

        images = [image.to(self.device) for image in images]
        outputs = model(images)
        
        for output in outputs:
            for key, value in output.items():
                output[key] = value.detach().cpu()

        return outputs, loss
    


        
    def _train_epoch(self, model, optimizer, loss_function, data_loader) -> float:
        train_loss = 0

        for images, targets in tqdm(data_loader, ncols=100):
            loss = self._train_batch(model, optimizer, loss_function, images, targets)

            # Encounter OOM
            if loss is None: continue

            train_loss += loss

        train_loss /= len(data_loader)
        return train_loss

    @torch.no_grad()
    def _val_epoch(self, model, loss_function, data_loader) -> Tuple[float, float]:
        val_loss = 0
        
        image_id_start = 1
        gt_images = []
        gt_annotations = []
        pred_annotations = []

        for images, targets in tqdm(data_loader, ncols=100):
            ret = self._val_batch(model, loss_function, images, targets)
            
            # OOM occured
            if ret is None: continue

            outputs, loss = ret
            
            batch_size = len(images)
            image_ids = list(range(image_id_start, image_id_start+batch_size))
            image_id_start += batch_size

            gt_id_start = len(gt_annotations) + 1

            gt_annotations.extend(
                targets_to_annotations(targets, image_ids, gt_id_start)
            )
            pred_annotations.extend(
                outputs_to_annotations(outputs, image_ids)
            )
            gt_images.extend([
                {
                    'id': image_id,
                    'width': image.shape[-1],
                    'height': image.shape[-2],
                }
                for image_id, image in zip(image_ids, images)
            ])

            val_loss += loss

        val_loss /= len(data_loader)

        ground_truth = to_ground_truth(gt_images, gt_annotations)
        ap50 = coco_ap50(ground_truth, pred_annotations)

        return ap50, val_loss

    def train(self,
              model: Detector,
              train_loader: DataLoader,
              val_loader: DataLoader,
              checkpoint_dir: str,
              max_epoches: int,
              optimizer: optim.Optimizer,
              scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
              loss_weights: Dict[str, float] = None,
              early_stop: bool = True):
        
        print(f'Train model by {self.device}')
        
        loss_weights = loss_weights or {'loss_classifier': 1.0,
                                        'loss_box_reg': 1.0,
                                        'loss_objectness': 1.0,
                                        'loss_rpn_box_reg': 1.0,
                                        'loss_mask': 1.0}
        
        loss_function = partial(self.weighted_loss, loss_weights=loss_weights)

        model = model.to(self.device)

        min_val_loss = float('inf')
        val_loss_increase_count = 0
        
        train_losses = []
        val_losses = []
        val_ap50s = []

        for epoch in range(max_epoches):
            train_loss = self._train_epoch(
                model,
                optimizer,
                loss_function,
                train_loader,
            )

            ap50, val_loss = self._val_epoch(
                model,
                loss_function,
                val_loader,
            )

            print(f'Epoch {epoch}')
            print(f'  Training loss: {train_loss:.5f}')
            print(f'  Validation loss: {val_loss:.5f}')
            print(f'  Validation AP@0.5: {ap50:.5f}')

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_ap50s.append(ap50)

            if scheduler: scheduler.step()

            model_path = f'{checkpoint_dir}/{model.model_name}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), model_path)

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                val_loss_increase_count = 0
            else:
                val_loss_increase_count += 1

            if val_loss_increase_count >= 2 and early_stop:
                print('Loss increased, training stopped.')
                break

        else:
            print('Max epoches reached.')

        print(f'Train losses: {train_losses}')
        print(f'Val losses: {val_losses}')
        print(f'Val AP@0.5: {val_ap50s}')