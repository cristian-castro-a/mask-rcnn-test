import json
import logging
from pathlib import Path
from typing import Tuple, List

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CustomMaskRCNN

DEVICE = torch.device('cpu') # torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 15
TMP = Path('tmp')

logger = logging.getLogger(__name__)

def evaluate(model: CustomMaskRCNN, val_loader: DataLoader, coco_gt: COCO, epoch: int) -> float:
    TMP.mkdir(parents=True, exist_ok=True)

    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Evaluating'):
            images = list(img.to(DEVICE) for img in images)
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target['image_id'].item())
                boxes =  output['boxes'].cpu()
                scores = output['scores'].cpu()
                labels = output['labels'].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box.tolist()
                    coco_results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [x_min, y_min, x_max-x_min, y_max-y_min],
                        'score': float(score)
                    })
        if not coco_results:
            return 0.

        # Save results to file for COCOeval
        results_path = str(TMP / f'results_epoch_{epoch}.json')

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(coco_results, f, indent=2)

        coco_dt = coco_gt.loadRes(results_path)
        coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]



def train_model(model: CustomMaskRCNN, train_loader: DataLoader, val_loader: DataLoader, coco: COCO) \
        -> Tuple[CustomMaskRCNN, List, List]:
    params =  [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.8, weight_decay=0.0001)

    # lr_schedule = lr_scheduler.StepLR(
    #     optimizer=optimizer,
    #     step_size=3,
    #     gamma=0.5
    # )

    loss_history = []
    map_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.

        for images, targets in tqdm(train_loader, desc=f'Epoch: {epoch+1}'):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # lr_schedule.step()

        average_loss = epoch_loss / len(train_loader)
        loss_history.append(average_loss)

        # Evaluate on validation set
        map_50_95 = evaluate(model, val_loader, coco, epoch+1)
        map_history.append(map_50_95)

        logger.info(f"[Epoch {epoch + 1}] Avg. Loss: {average_loss:.4f}")
        logger.info(f"[Epoch {epoch + 1}] mAP @ 0.50:0.95 = {map_50_95:.4f}")

    return model, loss_history, map_history
