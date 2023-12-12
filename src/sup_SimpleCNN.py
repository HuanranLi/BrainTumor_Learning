import torch
from torchvision import transforms, datasets
from SimpleCNN import *
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from CL import *
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils_SimpleCNN import *


import os
# Get the absolute path of the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Set the TORCH_HOME environment variable to the current directory
os.environ['TORCH_HOME'] = current_directory


def train_one_epoch(train_loader, model, device, criterion, optimizer):
    model.train()  # Set the fine-tuning model to training mode

    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)  # Calculate classification loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

    return running_loss


def evaluate_model(data_loader, model, device, criterion):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():  # No gradient computation for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return accuracy, avg_loss

sample_rate = (int(sys.argv[1]) + 1)/10
print(f'Sample Rate:', sample_rate)

# Default Hyperparameters
tune_hyperparams = {
    'batch_size': 32,
    'learning_rate': 0.01,
    'num_epochs': 200,
    'resize': (224, 224),
    'normalize_means': (0.5, 0.5, 0.5),
    'normalize_stds': (0.5, 0.5, 0.5),
    'model': 'sup_SimpleCNN',
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'train_dataset_dir': '../dataset/brain_tumor/Training',
    'test_dataset_dir': '../dataset/brain_tumor/Testing',
}
FT = {}
FT['train_loader'], FT['test_loader'] = load_data(tune_hyperparams, CL = False, sample_rate = sample_rate)
FT['model'], FT['device'] = setup_model(tune_hyperparams)
FT['optimizer'], FT['loss_function'] = init_optimizer_loss(tune_hyperparams, FT['model'])



# Classifier training loop
for epoch in range(tune_hyperparams['num_epochs']):
    train_loss = train_one_epoch(FT['train_loader'], FT['model'], FT['device'], FT['loss_function'], FT['optimizer'])
    print(f'Epoch {epoch}, Loss: {train_loss}')

    accuracy, avg_loss = evaluate_model(FT['test_loader'], FT['model'], FT['device'], FT['loss_function'])
    print(f'Evaluation Accuracy: {accuracy}%, Average Loss: {avg_loss}')
