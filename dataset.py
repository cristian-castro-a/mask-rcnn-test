from pathlib import Path
from typing import Callable, Optional, List

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from omegaconf import OmegaConf
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader


class ForksSpoonsKnifesDataset(Dataset):
    """
    Dataset class for the custom dataset of Forks-Spoons-Knifes.
    The data is in COCO Format
    """
    def __init__(self, images_dir: Path, annotations_path: Path, image_ids: List = None,
                 transforms: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        """
        :param images_dir: Directory containing the images.
        :param annotations_path: Path to the annotations in COCO format (.json file)
        :param transforms: Optional transform to apply on a sample.
        """
        self.images_dir = images_dir
        self.annotations_path = annotations_path
        self.coco = COCO(annotation_file=annotations_path)

        self.all_image_ids = list(self.coco.imgs.keys())
        self.image_ids = image_ids if image_ids is not None else self.all_image_ids

        self.transforms = transforms # Albumentations compose object
        self.cat_ids = self.coco.getCatIds(catNms=['fork', 'spoon', 'knife'])
        self.label_map = {cat_id: i + 1 for i, cat_id in enumerate(self.cat_ids)}

    def __len__(self):
        """
        :return: the total number of samples in the dataset
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Fetches the image and its corresponding annotation based on the index (idx).
        :param idx: Index of the sample to retrieve.
        :return: Tuple (image, target). The target is dictionary containing the bounding boxes, labels and image ID.
        """
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]  # Dictionary with info from the image
        image_path = self.images_dir / image_info['file_name']

        image = Image.open(image_path)
        if image.mode == 'P':
            image = image.convert('RGBA').convert('RGB')
        else:
            image = image.convert('RGB')
        image = np.array(image)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ids=ann_ids)

        boxes = []
        labels = []
        masks = []

        for annotation in annotations:
            # Bounding box
            bbox = annotation['bbox']
            x_min, y_min = bbox[0], bbox[1]
            x_max, y_max = x_min + bbox[2], y_min + bbox[3]
            boxes.append([x_min, y_min, x_max, y_max])

            # Label
            category_id = annotation['category_id']
            labels.append(self.label_map[category_id])

            # Mask
            mask = self.coco.annToMask(annotation) # Binary mask
            masks.append(mask)

        masks = np.stack(masks, axis=0) # Transforms the masks from list to ndarray

        if self.transforms:
            transformed = self.transforms(image=image,
                                          masks=list(masks),
                                          bboxes=boxes,
                                          class_labels=labels)
            image = transformed['image'].float() / 255.0
            masks = torch.stack([torch.as_tensor((m > 0.5), dtype=torch.float32) for m in transformed['masks']])
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.
            masks = torch.as_tensor(masks, dtype=torch.float32)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([image_id])
        area = torch.tensor([annotation['area'] for annotation in annotations], dtype=torch.float32)
        iscrowd = torch.tensor([annotation.get('iscrowd', 0) for annotation in annotations], dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        return image, target


def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def collate_fn(batch):
    return tuple(zip(*batch))


def get_loader(images_dir: Path, annotations_path: Path, ids: List, config: OmegaConf, train: bool = True) \
        -> DataLoader:
    if train:
        dataset = ForksSpoonsKnifesDataset(images_dir=images_dir,
                                           annotations_path=annotations_path,
                                           transforms=get_train_transforms(),
                                           image_ids=ids)
    else:
        dataset = ForksSpoonsKnifesDataset(images_dir=images_dir,
                                           annotations_path=annotations_path,
                                           transforms=A.Compose([
                                               A.Resize(512, 512),
                                               ToTensorV2()]),
                                           image_ids=ids)

    loader = DataLoader(dataset,
                        batch_size=config.train_config.batch_size,
                        shuffle=False,
                        collate_fn=collate_fn)

    return loader