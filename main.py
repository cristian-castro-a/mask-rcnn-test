import logging
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import OmegaConf
from pycocotools.coco import COCO

from dataset import get_loader
from model import CustomMaskRCNN
from train import train_model
from utils.utils import plot_training_curves, inference_and_visualization

DEVICE = torch.device('cpu') # torch.device("mps" if torch.backends.mps.is_available() else "cpu")

logger = logging.getLogger(__name__)


@hydra.main(config_path='conf', config_name='conf.yaml', version_base=None)
def run(config: OmegaConf) -> None:
    coco = COCO(annotation_file=config.annotations_path)
    all_image_ids = list(coco.imgs.keys())

    # Train and validation ids
    train_ids = all_image_ids[:76]
    val_ids = all_image_ids[76:]

    train_loader = get_loader(images_dir=Path(config.data_config.images_dir),
                              annotations_path=config.data_config.annotations_path,
                              ids=train_ids)
    val_loader = get_loader(images_dir=Path(config.data_config.images_dir),
                            annotations_path=config.data_config.annotations_path,
                            ids=val_ids)

    # Instantiate model
    model = CustomMaskRCNN(num_classes=4, config=config)
    model.to(DEVICE)

    # Get the trained model
    logger.info('Starting Training...')
    model, loss_history, mAP_history = train_model(model=model,
                                                   train_loader=train_loader,
                                                   val_loader=val_loader,
                                                   coco=coco,
                                                   config=config)

    # Save model
    logger.info("Saving model to tmp/model_weights.pth")
    torch.save(model.state_dict(), 'tmp/model_weights.pth')

    # Plot loss and mAP
    plot_training_curves(loss_history=loss_history, mAP_history=mAP_history)

    # # Load model
    # model.load_state_dict(torch.load('tmp/model_weights.pth', map_location=DEVICE))

    # Inference and visualization
    inference_and_visualization(model=model, val_loader=val_loader)


if __name__ == '__main__':
    """
    Toy example of transfer learning using Mask R-CNN
    Segmentations for: forks-knifes-spoons
    
    Cristian Castro
    """
    run()