import json
import logging
from pathlib import Path
from typing import Tuple, List

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from pycocotools import mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CustomMaskRCNN
from utils.utils import get_params

DEVICE = torch.device('cpu') # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TMP = Path('tmp')

logger = logging.getLogger(__name__)


def evaluate_map(model: CustomMaskRCNN, val_loader: DataLoader, coco_gt: COCO, epoch: int) -> float:
    TMP.mkdir(parents=True, exist_ok=True)

    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Evaluating'):
            images = [img.to(DEVICE) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target['image_id'].item())
                boxes =  output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()
                masks = output['masks'].cpu()

                for box, score, label, masks in zip(boxes, scores, labels, masks):
                    x_min, y_min, x_max, y_max = box.tolist()

                    binary_mask = (masks[0] > 0.5).numpy().astype(np.uint8)

                    rle = mask.encode(np.asfortranarray(binary_mask))
                    rle['counts'] = rle['counts'].decode('utf-8')

                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [x_min, y_min, x_max-x_min, y_max-y_min],
                        'score': float(score),
                        'segmentation': rle,
                    })
        if not coco_results:
            return 0.

        # Save results to file for COCOeval
        results_path = str(TMP / f'results_epoch_{epoch}.json')

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(coco_results, f, indent=2)

        coco_dt = coco_gt.loadRes(results_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]



def train_model(model: CustomMaskRCNN, train_loader: DataLoader, val_loader: DataLoader, coco: COCO, config: OmegaConf
                ) -> Tuple[CustomMaskRCNN, List, List]:

    mlflow.set_experiment(experiment_name=config.mlflow.experiment_name)
    mlflow.set_tracking_uri(uri=config.mlflow.uri)

    params =  [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=config.train_config.learning_rate,
                    momentum=config.train_config.momentum,
                    weight_decay=config.train_config.weight_decay)

    if config.train_config.scheduler.lr_scheduler:
        lr_schedule = lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config.train_config.scheduler.step_size,
            gamma=config.train_config.scheduler.gamma
        )

    # Dictionary to log into mlflow
    experiment_params = get_params(config=config)

    loss_history = []
    map_history = []

    with mlflow.start_run():
        mlflow.log_params(params=experiment_params)

        for epoch in range(config.train_config.epochs):
            model.train()
            epoch_loss = 0.
            classifier_loss = 0.
            box_reg_loss = 0.
            mask_loss = 0.

            for images, targets in tqdm(train_loader, desc=f'Epoch: {epoch+1}'):
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()
                classifier_loss += loss_dict['loss_classifier'].item()
                box_reg_loss += loss_dict['loss_box_reg'].item()
                mask_loss += loss_dict['loss_mask'].item()

            if config.train_config.scheduler.lr_scheduler:
                lr_schedule.step()

            average_loss = epoch_loss / len(train_loader)
            loss_history.append(average_loss)

            # Evaluate on validation set
            map_50_95 = evaluate_map(model, val_loader, coco, epoch+1)
            map_history.append(map_50_95)

            logger.info(f"[Epoch {epoch + 1}] Avg. Loss: {average_loss:.4f}")
            logger.info(f"[Epoch {epoch + 1}] mAP @ 0.50:0.95 = {map_50_95:.4f}")

            # Log metrics per epoch into mlflow
            mlflow.log_metric('total_avg_loss', average_loss, step=epoch)
            mlflow.log_metric('classifier_avg_loss', classifier_loss/len(train_loader), step=epoch)
            mlflow.log_metric('box_reg_avg_loss', box_reg_loss/len(train_loader), step=epoch)
            mlflow.log_metric('mask_avg_loss', mask_loss/len(train_loader), step=epoch)
            mlflow.log_metric('map_50_95', map_50_95, step=epoch)

    return model, loss_history, map_history
