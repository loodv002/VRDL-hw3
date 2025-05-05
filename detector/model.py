from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datetime import datetime

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        self.backbone = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
                                              trainable_backbone_layers=3,
                                              rpn_batch_size_per_image=1024,
                                              box_detections_per_img=1024)
        
        self.backbone.roi_heads.box_predictor = FastRCNNPredictor(
            1024, num_classes=5
        )
        self.backbone.roi_heads.mask_predictor = MaskRCNNPredictor(
            256, 256, num_classes=5
        )

        self.model_name = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    def forward(self, *args):
        return self.backbone(*args)
    