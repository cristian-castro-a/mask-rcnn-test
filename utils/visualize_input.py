from pathlib import Path
from dataset import ForksSpoonsKnifesDataset, get_loader
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

IMAGES_DIR = '../data/images'
ANNOTATIONS_PATH = '../data/annotations/instances.json'
CLASS_COLORS = {
    1: (255, 0, 0),    # Red for fork
    2: (0, 255, 0),    # Green for spoon
    3: (0, 0, 255),    # Blue for knife
}


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