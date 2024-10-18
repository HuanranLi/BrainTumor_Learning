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

This script performs the following steps:

	1.	Loads the datasets and applies transformations.
	2.	Initializes the feature extractor model using ResNet18.
	3.	Pre-trains the model with contrastive learning.
	4.	Fine-tunes the classifier on the brain tumor dataset.
	5.	Logs the results to TensorBoard.
