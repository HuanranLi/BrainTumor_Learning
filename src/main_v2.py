import torch
from torchvision import transforms, datasets
from SimpleCNN import *
import sys
import torch.optim as optim
from torch.utils.data import DataLoader
from CL import *
import torch.nn as nn
import torchvision.models as models

###################################
###### Train CL
###################################


base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 128

train_dataset = datasets.ImageFolder(root='./brain_tumor/Training', transform=base_transform)
train_loader_CL = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=CL_collate)

test_dataset = datasets.ImageFolder(root='./brain_tumor/Testing', transform=base_transform)
test_loader_CL = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=CL_collate)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.to(device)
num_ftrs = model.fc.in_features #512
model.fc = nn.Flatten()


optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = InfoNCELoss(temperature=1)
train_epochs = 5


model.train()
for epoch in range(train_epochs):
    running_loss = 0.0
    for i, (img1, img2) in enumerate(train_loader_CL):
        optimizer.zero_grad()
        embeddings1 = model(img1).to(device)
        embeddings2 = model(img2).to(device)

        loss = loss_function(embeddings1, embeddings2)  # Calculate contrastive loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{train_epochs} finished. Training Loss: {running_loss}')



###################################
###### Fine-Tune CL
###################################


train_dataset = datasets.ImageFolder(root='./brain_tumor/Training', transform=base_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root='./brain_tumor/Testing', transform=base_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classifier = nn.Linear(256, 4).to(device)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.1)
classification_criterion = nn.CrossEntropyLoss()

tune_epochs = 10

# Classifier training loop
classifier.train()
for epoch in range(tune_epochs):

    for images, labels in train_loader_FT:
        images, labels = images.to(device), labels.to(device)

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



###################################
###### Test
###################################

# Validation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in labeled_val_loader:
        images, labels = images.to(device), labels.to(device)

        representations = model(images)
        outputs = classifier(representations)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy of the model on the validation images: {accuracy * 100}%')
