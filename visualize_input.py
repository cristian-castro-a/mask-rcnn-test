from pathlib import Path
from dataset import ForksSpoonsKnifesDataset
import matplotlib.pyplot as plt
import numpy as np

IMAGES_DIR = 'data/images'
ANNOTATIONS_PATH = 'data/annotations/instances_default.json'
CLASS_COLORS = {
    1: (255, 0, 0),    # Red for fork
    2: (0, 255, 0),    # Green for spoon
    3: (0, 0, 255),    # Blue for knife
}


def run() -> None:
    dataset = ForksSpoonsKnifesDataset(images_dir=Path(IMAGES_DIR), annotations_path=Path(ANNOTATIONS_PATH))

    image, target = dataset[0]

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


if __name__ == '__main__':
    run()