from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torchvision.transforms.functional as F
import yaml
from omegaconf import OmegaConf
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks

from model import CustomMaskRCNN

DEVICE = torch.device('cpu') # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CLASS_NAMES = {1: 'fork', 2: 'spoon', 3: 'knife'}
CLASS_COLORS = {
    1: (255, 0, 0),    # Red for fork
    2: (0, 255, 0),    # Green for spoon
    3: (0, 0, 255),    # Blue for knife
}


def get_config(config: Path) -> Dict:
    with open(config, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def plot_training_curves(loss_history: List, mAP_history: List) -> None:
    epochs = [i for i in range(1, len(loss_history)+1)]

    fig = make_subplots(specs=[[{'secondary_y': True}]])

    fig.add_trace(go.Scatter(x=epochs, y=loss_history, mode='lines+markers', name='Loss'),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=epochs, y=mAP_history, mode='lines+markers', name='mAP'),
                  secondary_y=True)

    fig.update_layout(title_text='Training Metrics')
    fig.update_xaxes(title_text='Epochs')
    fig.update_yaxes(title_text='Loss', secondary_y=False)
    fig.update_yaxes(title_text='mAP', secondary_y=True)

    output_path = './tmp/training_metrics.html'
    fig.write_html(output_path)


def overlay_mask_on_image(image: torch.Tensor, mask: torch.Tensor, alpha: float = 0.5) -> np.ndarray:
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)

    mask_np = mask.cpu().numpy()
    mask_rgb = np.zeros_like(image_np)
    mask_rgb[:, :, 0] = mask_np * 255

    blended = np.clip(image_np * (1-alpha) + mask_rgb * alpha, 0 , 255).astype(np.uint8)

    return blended


def inference_and_visualization(model: CustomMaskRCNN, val_loader: DataLoader, config: OmegaConf) -> None:
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                image = images[i].detach().cpu()

                unnormalized = image.clone()
                # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                #
                # denormalized = unnormalized * std + mean

                plt.figure(figsize=(10, 10))
                plt.imshow(F.to_pil_image(unnormalized.cpu()))
                plt.axis('off')
                plt.savefig(f'tmp/original_{idx}_{i}.png', bbox_inches='tight')
                plt.close()

                keep = output['scores'].detach().cpu() > config.train_config.score_threshold
                masks = (output['masks'] > config.train_config.detection_threshold).detach().squeeze(1).cpu()[keep]
                labels = output['labels'].cpu()[keep]
                bboxes = output['boxes'][keep]

                if masks.numel() == 0:
                    continue

                # color_masks = torch.zeros((len(masks), *masks.shape[1:]), dtype=torch.bool)
                # for j, label in enumerate(labels):
                #     if label.item() == 0:  # Background
                #         continue
                #     color_masks[j] = masks[j]

                colors = [CLASS_COLORS[label.item()] for label in labels]
                image_with_masks = draw_segmentation_masks(
                    F.convert_image_dtype(unnormalized, dtype=torch.uint8),
                    masks=masks,
                    alpha=0.5,
                    colors=colors
                )

                plt.figure(figsize=(10, 10))
                plt.imshow(F.to_pil_image(image_with_masks))
                plt.axis('off')

                for index, label in enumerate(labels):
                    x_min, y_min, x_max, y_max = bboxes[index].tolist()
                    x_avg = (x_max + x_min) / 2
                    y_avg = (y_max + y_min) / 2
                    label = CLASS_NAMES[label.item()]
                    plt.text(x_avg, y_avg, label, size=25)

                plt.savefig(f'tmp/epochs_{config.train_config.epochs}_overlay_{idx}_{i}.png', bbox_inches='tight')
                plt.close()


def get_params(config: OmegaConf)-> Dict:
    """
    Transforms config from Hydra into a dictionary for mlflow
    """
    combined_config = {
        **{k: v for k, v in config.train_config.items() if k != 'scheduler'},
        **config.train_config.scheduler,
        **config.model_config
    }
    return combined_config