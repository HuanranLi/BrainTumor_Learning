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

from utils_main_v2 import *


import os
# Get the absolute path of the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Set the TORCH_HOME environment variable to the current directory
os.environ['TORCH_HOME'] = current_directory



def CL_train_one_epoch(model, train_loader, optimizer, loss_function, device):
    model.train()

    running_loss = 0.0
    for i, (img1, img2) in enumerate(train_loader):
        optimizer.zero_grad()
        embeddings1 = model(img1.to(device) )
        embeddings2 = model(img2.to(device) )

        loss = loss_function(embeddings1, embeddings2)  # Calculate contrastive loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss

def CL_evaluate_model(model, data_loader, loss_function, device):
    model.eval()

    running_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for i, (img1, img2) in enumerate(data_loader):
            embeddings1 = model(img1.to(device) )
            embeddings2 = model(img2.to(device) )

            loss = loss_function(embeddings1, embeddings2)  # Calculate contrastive loss
            running_loss += loss.item()

    return running_loss
learning_rate = 10**(-1 * int(sys.argv[1]))
print('Learning Rate:', learning_rate)

# Default Hyperparameters
train_hyperparams = {
    'batch_size': 128,
    'learning_rate': learning_rate,
    'num_epochs': 200,
    'resize': (224, 224),
    'normalize_means': (0.5, 0.5, 0.5),
    'normalize_stds': (0.5, 0.5, 0.5),
    'temperature': 1,
    'model': 'resnet18',
    'optimizer': 'Adam',
    'loss_function': 'InfoNCELoss',
    'train_dataset_dir': '../dataset/brain_tumor/Training',
    'test_dataset_dir': '../dataset/brain_tumor/Testing',
}

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(f'../logs/{current_time}')

# Model Init
CL = {}
CL['train_loader'], CL['test_loader'] = load_data(train_hyperparams, CL = True)
CL['model'], CL['device'] = setup_model(train_hyperparams)
CL['optimizer'], CL['loss_function'] = init_optimizer_loss(train_hyperparams, CL['model'])

# Training & Evaluation
for epoch in range(train_hyperparams['num_epochs']):
    print(f'Training epoch {epoch}')
    train_loss = CL_train_one_epoch(CL['model'],
                                    CL['train_loader'],
                                    CL['optimizer'],
                                    CL['loss_function'],
                                    CL['device'])
    print(f'Epoch {epoch+1} finished. Training Loss: {train_loss}')
    writer.add_scalar('Loss/Train', train_loss, epoch)

    evaluation_loss = CL_evaluate_model(CL['model'],
                                        CL['test_loader'],
                                        CL['loss_function'],
                                        CL['device'])
    print(f"Evaluation Loss: {evaluation_loss}")
    writer.add_scalar('Loss/Test', evaluation_loss, epoch)

print("Contrastive learning training complete.")



def FT_train_one_epoch(train_loader, CL_model, FT_model, device, criterion, optimizer):
    FT_model.train()  # Set the fine-tuning model to training mode

    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            representations = CL_model(images)

        outputs = FT_model(representations)
        loss = criterion(outputs, labels)  # Calculate classification loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

    return running_loss


def FT_evaluate_model(data_loader, CL_model, FT_model, device, criterion):
    FT_model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():  # No gradient computation for evaluation
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            representations = CL_model(images)
            outputs = FT_model(representations)

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


# Default Hyperparameters
tune_hyperparams = {
    'batch_size': 32,
    'learning_rate': 0.01,
    'num_epochs': 200,
    'resize': (224, 224),
    'normalize_means': (0.5, 0.5, 0.5),
    'normalize_stds': (0.5, 0.5, 0.5),
    'model': '3_layer_FFN',
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'train_dataset_dir': '../dataset/brain_tumor/Training',
    'test_dataset_dir': '../dataset/brain_tumor/Testing',
}
FT = {}
FT['train_loader'], FT['test_loader'] = load_data(tune_hyperparams, CL = False, sample_rate = 0.1)
FT['model'], FT['device'] = setup_model(tune_hyperparams)
FT['optimizer'], FT['loss_function'] = init_optimizer_loss(tune_hyperparams, FT['model'])



# Classifier training loop
for epoch in range(tune_hyperparams['num_epochs']):
    train_loss = FT_train_one_epoch(FT['train_loader'], CL['model'], FT['model'], FT['device'], FT['loss_function'], FT['optimizer'])
    print(f'Epoch {epoch}, Loss: {train_loss}')
    writer.add_scalar('FT_Loss/Train', train_loss, epoch)

    accuracy, avg_loss = FT_evaluate_model(FT['test_loader'], CL['model'], FT['model'], FT['device'], FT['loss_function'])
    print(f'Evaluation Accuracy: {accuracy}%, Average Loss: {avg_loss}')
    writer.add_scalar('FT_Loss/Test', avg_loss, epoch)
    writer.add_scalar('FT_Accuracy/Test', accuracy, epoch)


writer.close()
