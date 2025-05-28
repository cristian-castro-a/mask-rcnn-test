from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from dataset import ForksSpoonsKnifesDataset, get_loader

IMAGES_DIR = '../data/images'
ANNOTATIONS_PATH = '../data/annotations/instances.json'
CLASS_COLORS = {
    1: (255, 0, 0),    # Red for fork
    2: (0, 255, 0),    # Green for spoon
    3: (0, 0, 255),    # Blue for knife
}
CLASS_NAMES = {1: 'fork', 2: 'spoon', 3: 'knife'}
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def visualize_from_dataset(dataset: ForksSpoonsKnifesDataset) -> None:
    for i in range(2):
        image, target = dataset[i]

        image_np = image.permute(1, 2, 0).numpy() # Channels at the end
        image_np = (image_np * 255).astype('uint8')

        plt.figure(figsize=(10, 10))
        plt.imshow(image_np)
        plt.axis('off')

        for i in range(target['masks'].shape[0]):
            mask = target['masks'][i].numpy()
            label = target['labels'][i].item()
            color = CLASS_COLORS.get(label, (255, 255, 0))
            masked = mask.astype(bool)

            color_np = np.array(color, dtype=np.uint8)
            image_np[masked] = (image_np[masked] * 0.5 + color_np * 0.5).astype(np.uint8)

        plt.imshow(image_np.astype('uint8'))
        plt.show()


def visualize_from_dataloader(data_loader: DataLoader) -> None:
    for images, targets in iter(data_loader):
        for idx in range(2):
            image = images[idx]
            target = targets[idx]

            image_np = image.permute(1, 2, 0).numpy() # Channels at the end
            image_np = (image_np * 255).astype('uint8')
            image_np = np.clip(image_np, 0, 255).astype('uint8')

            plt.figure(figsize=(10, 10))

            for i in range(target['masks'].shape[0]):
                mask = target['masks'][i].numpy()
                label = target['labels'][i].item()
                color = CLASS_COLORS.get(label, (255, 255, 0))
                masked = mask.astype(bool)

                color_np = np.array(color, dtype=np.uint8)
                image_np[masked] = (image_np[masked] * 0.5 + color_np * 0.5).astype(np.uint8)

            plt.imshow(image_np)

            for i in range(target['boxes'].shape[0]):
                x_min, y_min, x_max, y_max = target['boxes'][i].tolist()
                x_avg = (x_max + x_min) / 2
                y_avg = (y_max + y_min) / 2
                text_img = (f'h/w: {(y_max-y_min)/(x_max-x_min):.2f}, label: {CLASS_NAMES[target["labels"][i].item()]}')

                plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor='black',
                                              facecolor='none', lw=4))
                plt.text(x_avg, y_avg, text_img, size=25)

            plt.show()


def run() -> None:
    dataset = ForksSpoonsKnifesDataset(images_dir=Path(IMAGES_DIR),
                                       annotations_path=Path(ANNOTATIONS_PATH))
    config = OmegaConf.create({'train_config': {'batch_size': 2}})
    data_loader = get_loader(images_dir=Path(IMAGES_DIR),
                             annotations_path=Path(ANNOTATIONS_PATH),
                             ids=[1,2],
                             config=config,
                             train=True)

    visualize_from_dataset(dataset=dataset)
    visualize_from_dataloader(data_loader=data_loader)




if __name__ == '__main__':
    run()