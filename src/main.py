import torch
from torchvision import transforms, datasets
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils import *
from models import *
from ContrastiveLearning import *
from FineTune import *

import os
# Get the absolute path of the current directory
current_directory = os.path.abspath(os.path.dirname(__file__))
# Set the TORCH_HOME environment variable to the current directory
os.environ['TORCH_HOME'] = current_directory

sample_rate = 0.1
print(f'Sample Rate:', sample_rate)

# Default Hyperparameters
train_hyperparams = {
    'batch_size': 128,
    'learning_rate': 1e-4,
    'num_epochs': 5,
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

# Default Hyperparameters
tune_hyperparams = {
    'batch_size': 32,
    'learning_rate': 0.01,
    'num_epochs': 5,
    'resize': (224, 224),
    'normalize_means': (0.5, 0.5, 0.5),
    'normalize_stds': (0.5, 0.5, 0.5),
    'model': '3_layer_FFN',
    'optimizer': 'Adam',
    'loss_function': 'CrossEntropyLoss',
    'train_dataset_dir': '../dataset/brain_tumor/Training',
    'test_dataset_dir': '../dataset/brain_tumor/Testing',
}

current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
writer = SummaryWriter(f'../logs/{current_time}')

def main():
    CL_main(train_hyperparams)
    TF_main(tune_hyperparams)

def CL_main(train_hyperparams):
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


def TF_main(tune_hyperparams):
    FT = {}
    FT['train_loader'], FT['test_loader'] = load_data(tune_hyperparams, CL = False, sample_rate = sample_rate)
    FT['model'], FT['device'] = setup_model(tune_hyperparams)
    FT['optimizer'], FT['loss_function'] = init_optimizer_loss(tune_hyperparams, FT['model'])



    # Classifier training loop
    for epoch in range(tune_hyperparams['num_epochs']):
        train_loss = FT_train_one_epoch(FT['train_loader'], CL['model'], FT['model'], FT['device'], FT['loss_function'], FT['optimizer'], CL['optimizer'])
        print(f'Epoch {epoch}, Loss: {train_loss}')
        writer.add_scalar('FT_Loss/Train', train_loss, epoch)

        accuracy, avg_loss = FT_evaluate_model(FT['test_loader'], CL['model'], FT['model'], FT['device'], FT['loss_function'])
        print(f'Evaluation Accuracy: {accuracy}%, Average Loss: {avg_loss}')
        writer.add_scalar('FT_Loss/Test', avg_loss, epoch)
        writer.add_scalar('FT_Accuracy/Test', accuracy, epoch)


    writer.close()

if __name__ == '__main__':
    main()
