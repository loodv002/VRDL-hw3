from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from contextlib import redirect_stdout
from io import StringIO
import numpy as np
import json
import os

from typing import Any, List, Dict

def coco_ap50(ground_truth: Dict[str, Any], prediction: List[Dict[str, Any]]) -> float:
    with open('.gt.tmp.json', 'w') as f:
        json.dump(ground_truth, f)
    with open('.pred.tmp.json', 'w') as f:
        json.dump(prediction, f)
    
    with redirect_stdout(StringIO()):
        coco_gt = COCO('.gt.tmp.json')
        coco_dt = coco_gt.loadRes('.pred.tmp.json')

        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.params.maxDets = [1, 100, 1000]
        coco_eval.params.recThrs = np.linspace(0.0, 1.0, 11)  
        coco_eval.params.useCats = 0
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap50 = coco_eval.stats[1]
    
    if os.path.exists('.gt.tmp.json'): os.remove('.gt.tmp.json')
    if os.path.exists('.pred.tmp.json'): os.remove('.pred.tmp.json')

    return ap50
