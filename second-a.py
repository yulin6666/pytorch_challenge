

from torchvision import transforms, datasets
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os

train_on_gpu = torch.cuda.is_available()
train_on_gpu




data_dir = 'flower_data'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_n = 0
valid_n = 0

for d in os.listdir(train_dir):
    train_n += len(os.listdir(train_dir + f'/{d}'))
    
for d in os.listdir(valid_dir):
    valid_n += len(os.listdir(valid_dir + f'/{d}'))
train_n, valid_n

# TODO: Define your transforms for the training and validation sets
image_transforms = {
        'train': transforms.Compose([
                 transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                 transforms.ToPILImage(),
                 transforms.TenCrop(size=224),
                 transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),   
        ]),
        'val': transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.ToPILImage(),
                transforms.TenCrop(size=224),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])
    }

batch_size = 16

# Datasets
data = {'train': datasets.ImageFolder(root=train_dir,
                                      transform=image_transforms['train']),
        'val': datasets.ImageFolder(root=valid_dir,
                                     transform=image_transforms['val'])
       }

dataloaders = {'train': DataLoader(data['train'], batch_size=batch_size),
              'val': DataLoader(data['val'], batch_size=batch_size)
              }


len(dataloaders['train'].dataset.samples)


# In[7]:


len(dataloaders['train'].dataset.classes)


# In[8]:


trainiter = iter(dataloaders['train'])
next(trainiter)[0].shape


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

name_to_cat = {name: cat for cat, name in cat_to_name.items()}
cat_to_name.keys()


# In[58]:


idx_to_name = {idx: cat_to_name[category] for category, idx in data['train'].class_to_idx.items()}
idx_to_name[10]


# In[10]:


for loader in dataloaders:
    class_to_idx = dataloaders[loader].dataset.class_to_idx 
    new_mapping = {cat_to_name.get(category): index for category, 
                   index in class_to_idx.items()}
    dataloaders[loader].dataset.class_to_idx = new_mapping
new_mapping['blanket flower']




model = models.vgg16(pretrained=True)
model.classifier[6]


# Freeze training for all layers
for param in model.parameters():
    param.requires_grad = False



n_inputs = model.classifier[6].in_features
n_classes = len(dataloaders['train'].dataset.classes)

# Classifier module
class Sequential(nn.Module):
    def __init__(self, n_inputs, n_classes, drop_prob=0.2):
        super(Sequential, self).__init__()
        # Fully connected layer
        self.fc1 = nn.Linear(n_inputs, int(n_inputs / 4))
        self.fc2 = nn.Linear(int(n_inputs / 4), int(n_inputs / 8))
        self.fc3 = nn.Linear(int(n_inputs / 8), n_classes)
        
        # Dropout
        self.dropout = nn.Dropout(p=drop_prob)
        
        # Output layer
        self.out = nn.LogSoftmax(dim = 1)
    
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.out(x)
        return x

classifier = Sequential(n_inputs, n_classes)
model.classifier[6] = nn.Linear(n_inputs, n_classes)


# In[47]:


model.classifier


# In[48]:


# TODO: Build and train your network
if train_on_gpu:
    model = model.to('cuda')
from torchsummary import summary

summary(model, input_size = (3, 224, 224), batch_size = 1024)



for param in model.parameters():
    if param.requires_grad:
        print(param.shape)


# In[50]:


pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params


# In[51]:


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_total_params


# In[52]:


optimizer = optim.Adam(model.parameters())




n_epochs = 20
max_epochs_stop = 3

save_file_name = 'second_aug.pt'





def train(model, train_loader, valid_loader, save_file_name,
          max_epochs_stop=3,
          n_epochs=20):

    # Early stopping details
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    # specify loss function
    criterion = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters())
    
    try:
        print(f'Current training epochs:{model.epochs}.')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.')
        
    for epoch in range(n_epochs):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        ###################
        # train the model #
        ###################
        model.train()

        for ii, (data, target) in enumerate(train_loader):
            bs, ncrops, c, h, w = data.size()
            
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            # Fuse batch size and the number of crops
            output = model(data.view(-1, c, h, w))

            # Take average over the crops
            output = output.view(bs, ncrops, -1).mean(1)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

            # Calculate accuracy
            _, pred = torch.max(output, dim = 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            train_acc += accuracy.item()

            print(
                f'Epoch: {epoch} \t {100 * ii / len(train_loader):.2f}% complete.', end='\r')
        
        else:
            model.epochs += 1
            with torch.no_grad():
                model.eval()
                # Validation loop
                for data, target in valid_loader:
                    bs, ncrops, c, h, w = data.size()
                    
                    # move tensors to GPU if CUDA is available
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    output = model(data.view(-1, c, h, w))
                    # Take average over the crops
                    output = output.view(bs, ncrops, -1).mean(1)
                    # calculate the batch loss
                    loss = criterion(output, target)
                    # update average validation loss
                    valid_loss += loss.item()

                    # Calculate accuracy
                    _, pred = torch.max(output, dim = 1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    valid_acc += accuracy.item()

                # calculate average losses
                train_loss = train_loss/len(train_loader)
                valid_loss = valid_loss/len(valid_loader)

                train_acc = train_acc/len(train_loader)
                valid_acc = valid_acc/len(valid_loader)

                # print training/validation statistics
                print('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                    epoch, train_loss, valid_loss))
                print(
                    f'Training Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%')

                # save model if validation loss has decreased
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                        valid_loss_min,
                        valid_loss))
                    torch.save(model.state_dict(), save_file_name)
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                else:
                    epochs_no_improve += 1
                    print(f'{epochs_no_improve} epochs with no improvement.')
                    if epochs_no_improve >= max_epochs_stop:
                        print('Early Stopping')
                        break
# In[62]:


train(model, dataloaders['train'], dataloaders['val'], max_epochs_stop=10,
      save_file_name=save_file_name, n_epochs = 50)

def save_checkpoint(model, optimizer, path):
    checkpoint = {
        'class_to_idx': data['train'].class_to_idx,
        'idx_to_name': idx_to_name,
        'epochs': model.epochs,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    
    torch.save(checkpoint, path)
    
save_checkpoint(model, optimizer, 'vgg16-aug.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, model):
    # Make sure to set parameters as not trainable
    for param in model.parameters():
        param.requires_grad = False
        
    # Load in checkpoint
    checkpoint = torch.load(filepath)
    # Extract classifier
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_name = checkpoint['idx_to_name']
    model.epochs = checkpoint['epochs']
    
    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    if train_on_gpu:
        model = model.to('cuda')
    
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} total gradient parameters.')
    
    return model, optimizer

model, optimizer = load_checkpoint('vgg16-aug.pth', models.vgg16(pretrained=True))
print(model)
