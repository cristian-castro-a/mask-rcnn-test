import torch.nn as nn
from omegaconf import OmegaConf
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class CustomMaskRCNN(nn.Module):
    def __init__(self, num_classes: int, config: OmegaConf, pretrained_backbone: bool = True):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = resnet_fpn_backbone(backbone_name='resnet50',
                                       weights=weights,
                                       trainable_layers=config.model_config.trainable_layers
                                       )
        # # Transfer learning
        # for param in backbone.parameters():
        #     param.requires_grad = False

        # FPN has 5 feature maps
        anchor_generator = AnchorGenerator(
            sizes=tuple((i,) for i in config.model_config.anchor_sizes),
            aspect_ratios=(tuple(i for i in config.model_config.anchor_aspect_ratios),)*5
        )

        # Instantiate Mask R-CNN
        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator
        )

        # Custom box prediction
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Custom mask prediction
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = config.model_config.hidden_layers_mask
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)