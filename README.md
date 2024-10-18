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

### Installation
1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/brain-tumor-classification.git
   cd brain-tumor-classification

	2.	Install the required packages:

pip install -r requirements.txt


	3.	Set up your dataset:
	•	Place your training data in ../dataset/brain_tumor/Training.
	•	Place your testing data in ../dataset/brain_tumor/Testing.

Directory Structure

brain-tumor-classification/
│
├── models/               # Custom model architectures
├── utils/                # Utility functions for data loading, augmentation, etc.
├── ContrastiveLearning.py # Contrastive learning methods
├── FineTune.py           # Fine-tuning and evaluation functions
├── train.py              # Main training script
└── README.md             # Project documentation

Running the Project

To run the training and evaluation process, execute:

python train.py

This script performs the following steps:

	1.	Loads the datasets and applies transformations.
	2.	Initializes the feature extractor model using ResNet18.
	3.	Pre-trains the model with contrastive learning.
	4.	Fine-tunes the classifier on the brain tumor dataset.
	5.	Logs the results to TensorBoard.

Hyperparameters

The training and fine-tuning hyperparameters can be modified in the script. Default settings include:

	•	Contrastive Learning:
	•	Batch size: 128
	•	Learning rate: 1e-4
	•	Model: ResNet18
	•	Loss: InfoNCE
	•	Fine-Tuning:
	•	Batch size: 32
	•	Learning rate: 0.01
	•	Model: 3-layer feedforward network
	•	Loss: CrossEntropy

TensorBoard

Monitor the training progress using TensorBoard. Run the following command:

tensorboard --logdir=../logs

Open http://localhost:6006 in your browser to view the logs.

Customization

	•	Modify the model architecture by editing the files in models/ and the setup_model() function in train.py.
	•	Change the dataset directories by adjusting the train_dataset_dir and test_dataset_dir parameters in the hyperparameter dictionary.
	•	Adjust the sample rate for fine-tuning by modifying the sample_rate in the main() function.

Contributing

Feel free to submit issues or pull requests for improvements.

License

This project is licensed under the MIT License.

Acknowledgments

	•	PyTorch for the deep learning framework.
	•	TensorBoard for logging and visualization.
