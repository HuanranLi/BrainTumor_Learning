
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from CL import *


# Define a 3-layer Feedforward Neural Network
class ThreeLayerFFN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ThreeLayerFFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



# Load and preprocess data
def load_data(hyperparams, CL = False):
    print("Loading and preprocessing data...")

    base_transform = transforms.Compose([
        transforms.Resize(hyperparams['resize']),
        transforms.ToTensor(),
        transforms.Normalize(hyperparams['normalize_means'], hyperparams['normalize_stds'])
    ])

    train_dataset = datasets.ImageFolder(root=hyperparams['train_dataset_dir'], transform=base_transform)
    if CL:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, collate_fn=CL_collate)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    test_dataset = datasets.ImageFolder(root=hyperparams['test_dataset_dir'], transform=base_transform)
    if CL:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=True, collate_fn=CL_collate)
    else:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=True)


    print("Data loaded and preprocessed.")
    return train_loader, test_loader


def setup_model(hyperparams):
    print("Setting up the model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_name = hyperparams['model']
    if model_name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Flatten()
    elif model_name == '3_layer_FFN':
        # Define input_size, hidden_size, and num_classes as per your requirements
        input_size = 512  # Example value
        hidden_size = 256  # Example value
        num_classes = 4  # Example value
        model = ThreeLayerFFN(input_size, hidden_size, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported.")

    model.to(device)
    print(f"Model {model_name} setup complete. Model moved to device: {device}")
    return model, device



# Initialize optimizer and loss function
def init_optimizer_loss(hyperparams, model):
    optimizer_name = hyperparams['optimizer']
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    # Add more optimizers here as elif conditions
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")

    loss_function_name = hyperparams['loss_function']
    if loss_function_name == 'InfoNCELoss':
        loss_function = InfoNCELoss(temperature=hyperparams['temperature'])
    elif loss_function_name == 'CrossEntropyLoss':
        loss_function = nn.CrossEntropyLoss()
    # Add more loss functions here as elif conditions
    else:
        raise ValueError(f"Loss function {loss_function_name} not supported.")

    return optimizer, loss_function
