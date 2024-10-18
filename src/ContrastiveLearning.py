import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from PIL import Image


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

def Random_transform(inputs):
    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=None, shear=(10, 20)),  # Affine transformation with shear
        transforms.RandomRotation(degrees=15),  # Rotation
        transforms.ToTensor(),
    ])

    augmented_inputs = torch.stack([augmentation(img) for img in inputs])
    return augmented_inputs


def CL_collate_v2(batch):
    input_data_list, label_list = zip(*batch)

    # Concatenate input data along the batch dimension
    augmented_inputs1 = Random_transform(input_data_list)
    augmented_inputs2 = Random_transform(input_data_list)


    return augmented_inputs1, augmented_inputs2
