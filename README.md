# 🍴 Mask R-CNN on Custom Dataset (Forks, Spoons, Knives)

This is a toy project designed to explore the architecture and practical implementation of **Mask R-CNN** using **PyTorch**, **Hydra**, and **MLflow** on a custom dataset of kitchen utensils: forks, spoons, and knives. The main goal was to gain hands-on experience with instance segmentation, pipeline debugging, and modern ML engineering practices.

---

## Project Highlights

- 📸 **Custom Dataset**: 80 annotated images (COCO format) of forks, spoons, and knives.
- 🏗️ **Model Architecture**: Mask R-CNN with a ResNet-FPN backbone via `torchvision`.
- 🔄 **Transfer Learning**: Leveraged pre-trained backbones for faster convergence.
- 🧪 **Sanity Check**: Overfit on 2 images to validate the pipeline integrity.
- 💾 **Training**: Trained on CPU with 76 images, validated on 4 (MacBook Pro M1 was crashing when trying to use 'mps').
- ⚙️ **Modular Design**: Configurable with Hydra, tracked with MLflow.
- 🧹 **Clean Engineering**: Project structured for reproducibility and maintainability.

---

## Results

- The model was able to overfit on 2 images successfully — confirming pipeline correctness.
- On the full (limited) dataset, training and validation were completed locally using CPU.
- Performance was not the focus due to dataset size, but results were qualitatively acceptable and valuable for architectural understanding.

---

## Disclaimer
This is a toy project intended for educational purposes. Performance is not generalizable due to the small dataset size and lack of GPU-based training.