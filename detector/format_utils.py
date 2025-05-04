from typing import Any, Dict, List

from .image_utils import encode_mask

def outputs_to_annotations(outputs: List[Dict[str, Any]], image_ids: List[int]) -> List[Dict[str, Any]]:
    annotations = []

    for image_id, output in zip(image_ids, outputs):
        for box, score, label, mask in zip(output['boxes'], 
                                           output['scores'], 
                                           output['labels'], 
                                           output['masks']):
            
            if label == 0: continue

            box = box.detach().cpu().numpy().tolist()
            score = score.detach().cpu().item()
            label = label.detach().cpu().item()
            mask = mask[0].detach().cpu()
            box[2] -= box[0]
            box[3] -= box[1]

            segmentation = encode_mask(mask.numpy() > 0.5)

            annotations.append({
                'image_id': image_id,
                'bbox': box,
                'score': score,
                'category_id': label,
                'segmentation': segmentation,
            })
    return annotations

def targets_to_annotations(targets: List[Dict[str, Any]], image_ids: List[int], id_start: int) -> List[Dict[str, Any]]:
    annotations = []

    for image_id, target in zip(image_ids, targets):
        for box, label, mask in zip(target['boxes'], 
                                    target['labels'], 
                                    target['masks']):
            
            box = box.detach().cpu().numpy().tolist()
            label = label.detach().cpu().item()
            mask = (mask.detach().cpu() != 0)
            box[2] -= box[0]
            box[3] -= box[1]

            segmentation = encode_mask(mask.numpy())
            area = mask.sum().item()

            annotations.append({
                'id': 1,
                'image_id': image_id,
                'category_id': label,
                'bbox': box,
                'area': area,
                'iscrowd': 0,
                'segmentation': segmentation
            })

    for id, annotation in enumerate(annotations, id_start):
        annotation['id'] = id

    return annotations

def to_ground_truth(images: List[Dict[str, Any]],
                    annotations: List[Dict[str, Any]]):
    
    categories = [
        {'id': i}
        for i in range(1, 5)
    ]
    
    return {
        'images': images,
        'annotations': annotations,
        'categories': categories,
    }