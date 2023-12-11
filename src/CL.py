import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image

class ContrastiveCNN(nn.Module):
    def __init__(self):
        super(ContrastiveCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Inside your __init__ method after defining the conv and pool layers
        with torch.no_grad():
            # Passing a dummy input through the conv and pool layers to calculate output size
            dummy_input = torch.zeros(1, 3, 224, 224)  # Batch size of 1, 3 color channels, 224x224 image size
            dummy_output = self.pool(self.conv2(self.pool(self.conv1(dummy_input))))
            # The number of features is the product of the dimensions of the output from the conv layers
            self.flattened_feature_size = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)


            # Now define the rest of your layers using the calculated flattened_feature_size
            self.fc1 = nn.Linear(self.flattened_feature_size, 512)
            self.projection_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = x.view(x.size(0), -1)
        # Make sure the flattened tensor has the correct size
        assert x.size(1) == self.flattened_feature_size, f"Expected size {self.flattened_feature_size}, but got {x.size(1)}"
        # Pass the flattened features through the first fully connected layer
        x = F.relu(self.fc1(x))
        # Pass the features through the projection head
        embeddings = self.projection_head(x)
        return embeddings

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)

        if torch.isnan(z_i).any() or torch.isinf(z_i).any():
            raise ValueError("NaN/inf values detected in img1")
        if torch.isnan(z_j).any() or torch.isinf(z_j).any():
            raise ValueError("NaN/inf values detected in img2")

        # Cosine similarity as dot product between normalized vectors
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)


        # Create the labels for the positive pairs (1s for positive, 0s for negative)
        batch_size = z_i.shape[0]
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        labels = torch.cat((labels, labels), dim=0)

        # Mask to ignore self-contrast cases (diagonal elements in the similarity matrix)
        mask = torch.eye(2*batch_size, dtype=torch.bool).to(similarity_matrix.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        softmax_similarity_matrix = F.softmax(similarity_matrix, dim=1)

        # Compute the InfoNCE loss
        # Using log-softmax for numerical stability
        logits = softmax_similarity_matrix / self.temperature

        # Check for NaN values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN/inf values detected in logits")

        loss = F.cross_entropy(logits, labels, reduction='sum') / (2 * batch_size)

        return loss

class ContrastiveTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        if isinstance(x, tuple):
            print(f"Type of x[0] before transform: {type(x[0])}")
            img1 = self.base_transform(x[0])
            img2 = self.base_transform(x[1])
            print(f"Type of img1 after transform: {type(img1)}")
        else:
            print(f"Type of x before transform: {type(x)}")
            img1 = self.base_transform(x.copy())
            img2 = self.base_transform(x.copy())
            print(f"Type of img1 after transform: {type(img1)}")
        return img1, img2

def RandomCrop_data(inputs):

    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    augmented_inputs = torch.stack([augmentation(img) for img in inputs])
    return augmented_inputs

def H_flip(inputs):
    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    augmented_inputs = torch.stack([augmentation(img) for img in inputs])
    return augmented_inputs

def GaussBlur_data(inputs):

    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])

    augmented_inputs = torch.stack([augmentation(img) for img in inputs])
    return augmented_inputs

def CL_collate(batch):
    input_data_list, label_list = zip(*batch)

    # Concatenate input data along the batch dimension
    augmented_inputs1 = RandomCrop_data(input_data_list)
    augmented_inputs2 = GaussBlur_data(input_data_list)


    return augmented_inputs1, augmented_inputs2


def CL_collate_v2(batch):
    input_data_list, label_list = zip(*batch)

    # Concatenate input data along the batch dimension
    augmented_inputs1 = RandomCrop_data(input_data_list)
    augmented_inputs2 = H_flip(input_data_list)


    return augmented_inputs1, augmented_inputs2
