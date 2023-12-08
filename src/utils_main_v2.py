
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from CL import *
import random


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


def sample_dataset(dataset, percentage):
    # Ensure the percentage is between 0 and 1
    percentage = max(0, min(percentage, 1))

    # Get class indices
    class_to_indices = dataset.class_to_idx
    indices_to_class = {v: k for k, v in class_to_indices.items()}

    # Sample data
    sampled_indices = []
    for class_idx in class_to_indices.values():
        class_indices = [i for i, label in enumerate(dataset.targets) if label == class_idx]
        k = int(len(class_indices) * percentage)
        sampled_indices.extend(random.sample(class_indices, k))

    # Subset dataset
    sampled_dataset = torch.utils.data.Subset(dataset, sampled_indices)

    return sampled_dataset

# Example Usage
# dataset = ImageFolder(root=hyperparams['train_dataset_dir'], transform=base_transform)
# sampled_dataset = sample_dataset(dataset, 0.2)  # Sample 20% of data from each class



# Load and preprocess data
def load_data(hyperparams, CL = False, sample_rate = 1):
    print("Loading and preprocessing data...")

    base_transform = transforms.Compose([
        transforms.Resize(hyperparams['resize']),
        transforms.ToTensor(),
        transforms.Normalize(hyperparams['normalize_means'], hyperparams['normalize_stds'])
    ])

    train_dataset = datasets.ImageFolder(root=hyperparams['train_dataset_dir'], transform=base_transform)
    if sample_rate < 1:
        train_dataset = sample_dataset(train_dataset, sample_rate)
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
    elif model_name == 'sup_resnet18':
        # Modify the final fully connected layer
        #model = models.resnet18(pretrained=True)
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 4)  # 4 classes
    elif model_name == '3_layer_FFN':
        input_size = 512
        hidden_size = 256
        num_classes = 4
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
