import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

ANNOTATIONS_PATH = '../data/annotations/instances_default.json'


def run() -> None:
    coco = COCO(ANNOTATIONS_PATH)

    aspect_ratios = []
    areas =  []

    for annotation in coco.dataset['annotations']:
        segmentation = annotation['segmentation']

        if not segmentation:
            continue

        # Get the statistics
        bbox = annotation['bbox']
        width, height = bbox[2], bbox[3]

        if height > 0:
            aspect_ratio = width / height
            aspect_ratios.append(aspect_ratio)

            area = width*height
            areas.append(area)

    aspect_ratios = np.array(aspect_ratios)
    scales = np.sqrt(areas)

    print("Mean h/w:", np.mean(aspect_ratios))
    print("Median h/w:", np.median(aspect_ratios))
    print("Min h/w:", np.min(aspect_ratios))
    print("Max h/w:", np.max(aspect_ratios))

    print("Mean scales:", np.mean(scales))
    print("Median scales:", np.median(scales))
    print("Min scales:", np.min(scales))
    print("Max scales:", np.max(scales))

    plt.hist(aspect_ratios, bins=15)
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Frequency')
    plt.title('Aspect Ratio Distribution')
    plt.savefig('tmp/histogram.png', bbox_inches='tight')


if __name__ == '__main__':
    """
    Explores aspect ratios from the annotations -> Input for transfer learning
    """
    run()