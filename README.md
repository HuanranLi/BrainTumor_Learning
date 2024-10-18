# Brain Tumor Classification with Contrastive Learning and Fine-Tuning

This repository implements a pipeline for brain tumor image classification using a combination of contrastive learning (self-supervised learning) and fine-tuning techniques. The project utilizes PyTorch and various utility modules for handling datasets, models, and training loops.

## Features
- **Contrastive Learning**: Pre-training of the feature extractor using self-supervised contrastive learning (InfoNCE Loss).
- **Fine-Tuning**: Training a classification head on a limited labeled dataset to predict tumor classes.
- **Data Augmentation**: Applies various transformations for robust training.
- **Logging**: Uses TensorBoard to log training and evaluation metrics.

## Setup Instructions

### Prerequisites
- Python 3.8+
- PyTorch
- torchvision
- TensorBoard for visualizations

### Running the Project

To run the training and evaluation process, execute:

python main.py
