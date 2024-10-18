import torch

def FT_train_one_epoch(train_loader, CL_model, FT_model, device, criterion, optimizer, CL_optimizer):
    FT_model.train()  # Set the fine-tuning model to training mode

    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            representations = CL_model(images)

        outputs = FT_model(representations)
        loss = criterion(outputs, labels)  # Calculate classification loss

        optimizer.zero_grad()
        #CL_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #CL_optimizer.step()

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
