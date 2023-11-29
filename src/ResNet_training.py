import torch
from torchvision import transforms, datasets
from SimpleCNN import *
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from CL import *
import torch.nn as nn
import torchvision.models as models

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
contrastive_transform = ContrastiveTransform(base_transform)

batch_size = 128

train_dataset = datasets.ImageFolder(root='./brain_tumor/Training', transform=base_transform)
train_loader_FT = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CL_collate)

validation_transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Assuming you have a validation dataset in a directory './brain_tumor/Validation'
val_dataset = datasets.ImageFolder(root='./brain_tumor/Testing', transform=validation_transform)

# Create the validation data loader
labeled_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features #512
model.fc = nn.Flatten()
model.train()  # Set the model to training mode
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = InfoNCELoss(temperature=1)
num_epochs = 5

    # Contrastive training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (img1, img2) in enumerate(train_loader):
        optimizer.zero_grad()
        embeddings1 = model(img1)
        embeddings2 = model(img2)

        # # Check for NaN values
        # if torch.isnan(embeddings1).any() or torch.isnan(embeddings2).any():
        #     raise ValueError("NaN values detected in embeddings")

        # # Check for Inf values
        # if torch.isinf(embeddings1).any() or torch.isinf(embeddings2).any():
        #     raise ValueError("Inf values detected in embeddings")

        loss = loss_function(embeddings1, embeddings2)  # Calculate contrastive loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs} finished. Training Loss: {running_loss}')

# After contrastive training, initialize the classifier
model.eval()  # Set the contrastive model to evaluation mode
# with torch.no_grad():
#     representations = model(images)
#     print("Representations shape:", representations.shape)

classifier = nn.Linear(256, 4).to('cpu')
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
classification_criterion = nn.CrossEntropyLoss()

# Classifier training loop
for epoch in range(num_epochs):
    # Assume labeled_train_loader is a DataLoader that provides (image, label) pairs
    for images, labels in train_loader_FT:
        images, labels = images.to('cpu'), labels.to('cpu')

        # Forward pass to get representations
        with torch.no_grad():
            representations = model(images)

        # Forward pass through the classifier
        outputs = classifier(representations)
        loss = classification_criterion(outputs, labels)  # Calculate classification loss

        # Backward and optimize
        classifier_optimizer.zero_grad()
        loss.backward()
        classifier_optimizer.step()

# Validation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in labeled_val_loader:
        images, labels = images.to('cpu'), labels.to('cpu')
        representations = model(images)
        outputs = classifier(representations)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the validation images: {accuracy * 100}%')