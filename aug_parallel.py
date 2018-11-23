from torchvision import transforms, datasets
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

import numpy as np

import os
import json

from torchsummary import summary

from timeit import default_timer as timer


train_on_gpu = torch.cuda.is_available()
print(f'Training on gpu: {train_on_gpu}')


data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_n = 0
valid_n = 0

train_n = 0
valid_n = 0

for d in os.listdir(train_dir):
    train_n += len(os.listdir(train_dir + f'/{d}'))

for d in os.listdir(valid_dir):
    valid_n += len(os.listdir(valid_dir + f'/{d}'))

print(f'Training images: {train_n} Validation images: {valid_n}')

image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.ToPILImage(),
        transforms.TenCrop(size=224),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops]))
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

batch_size = 32

# Datasets
data = {'train': datasets.ImageFolder(root=train_dir,
                                      transform=image_transforms['train']),
        'val': datasets.ImageFolder(root=valid_dir,
                                    transform=image_transforms['val'])
        }

dataloaders = {'train': DataLoader(data['train'], batch_size=batch_size),
               'val': DataLoader(data['val'], batch_size=batch_size)
               }

trainiter = iter(dataloaders['train'])
print(f'Training shape :{next(trainiter)[0].shape}')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

name_to_cat = {name: cat for cat, name in cat_to_name.items()}

class_to_idx = data['train'].class_to_idx
idx_to_name = {idx: cat_to_name[category]
               for category, idx in data['train'].class_to_idx.items()}

model = models.vgg16(pretrained=True)
model.classifier[6]

n_inputs = model.classifier[6].in_features
n_classes = len(dataloaders['train'].dataset.classes)


def load_checkpoint(filepath, model):
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False

    # Load in checkpoint
    checkpoint = torch.load(filepath)

    # Extract classifier
    model.classifier = checkpoint['classifier']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    if train_on_gpu:
        model = model.to('cuda')

    model.cat_to_name = checkpoint['cat_to_name']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_name = checkpoint['idx_to_name']
    model.epochs = checkpoint['epochs']

    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel()
                                 for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')
    print(f'Model has been trained for {model.epochs} epochs.')

    return model, optimizer


model, optimizer = load_checkpoint(
    'vgg16.pth', model)
summary(model, input_size=(3, 224, 224), batch_size=batch_size)

print(f'Model classifier: {model.classifier}')

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f'Training on {torch.cuda.device_count} gpus.')

for param in model.parameters():
    if param.requires_grad:
        print(param.shape)


def train(model, criterion, optimizer, train_loader, valid_loader, save_file_name,
          max_epochs_stop=3, n_epochs=20):

    # Early stopping details
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    # Number of epochs already trained
    try:
        print(f'Current training epochs: {model.epochs}.')
    except Exception as e:
        model.epochs = 0
        print(f'Starting Training from Scratch.')

    overall_start = timer()
    # Iterate through epochs
    for epoch in range(n_epochs):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        model.train()

        start = timer()
        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            bs, crops, c, h, w = data.size()

            # Tensors on gpu
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()

            # Predicted outputs
            # outputs are not probabilities
            output = model(data.view(-1, c, h, w))
            output = output.view(bs, crops, -1).mean(1)

            # Loss and backpropagation
            loss = criterion(output, target)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss
            train_loss += loss.item()

            # Calculate accuracy by finding max probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item()

            # Track training
            print(
                f'Epoch: {epoch}\t{100 * ii / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed.', end='\r')

        # After training loops ends
        else:
            model.epochs += 1
            model.eval()
            # Don't need to keep track of gradients
            with torch.no_grad():

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)
                    # Validation loss
                    loss = criterion(output, target)
                    valid_loss += loss.item()

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    valid_acc += accuracy.item()

                # Calculate average losses
                train_loss = train_loss / len(train_loader)
                valid_loss = valid_loss / len(valid_loader)
                # Calculate average accuracy
                train_acc = train_acc / len(train_loader)
                valid_acc = valid_acc / len(valid_loader)

                # Print training and validation results
                print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch, train_loss, valid_loss))
                print(
                    f'Training Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min - 0.01:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    print(f'{epochs_no_improve} epochs with no improvement.')
                    if epochs_no_improve >= max_epochs_stop:
                        print('Early Stopping')
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')
                        break


criterion = nn.CrossEntropyLoss()
train(model, criterion, optimizer,
      dataloaders['train'], dataloaders['val'], max_epochs_stop=10,
      save_file_name='aug-parallel.pt', n_epochs=50)


def save_checkpoint(model, optimizer, path, save_cpu=False):
    if save_cpu:
        model = model.to('cpu')
        path = path.split('.')[0] + '-cpu.pth'

    checkpoint = {
        'cat_to_name': cat_to_name,
        'class_to_idx': data['train'].class_to_idx,
        'idx_to_name': idx_to_name,
        'epochs': model.epochs,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(checkpoint, path)


save_checkpoint(model, optimizer, 'vgg16-aug.pth')
model, optimizer = load_checkpoint(
    'vgg16-aug-parallel.pth', models.vgg16(pretrained=True))
print(model)
