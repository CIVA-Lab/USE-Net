import torch
import torch.optim as optim
import numpy as np
import torchvision
import glob
import torch.hub

from torchvision import models
from torchsummary import summary
from torch.optim import lr_scheduler

from data.dataLoader import (USENetPathLoader, USENetDataLoader)
from nets.USENet import USENet
from trainer.trainer import trainModel

# inputs, labels and markers path
folderData = sorted(glob.glob("./dataset/augTrainData/SPCN/*.png"))
folderMask = sorted(glob.glob("./dataset/augTrainData/BinaryMask/*.png")) 
folderMarker = sorted(glob.glob("./dataset/augTrainData/Marker/*.png")) 

print(len(folderData))
print(len(folderMask))
print(len(folderMarker))

# model name
modelName = 'USENet'

dataset = USENetPathLoader(folderData, folderMask, folderMarker)

trLengths = int(len(dataset)*0.9);
lengths = [trLengths, len(dataset) - trLengths]

train_dataset, val_dataset = torch.utils.data.random_split(dataset, lengths)

train_data = USENetDataLoader(train_dataset)
val_data = USENetDataLoader(val_dataset)

print(len(train_data))
print(len(val_data))

# set batch size
batchSize = 8

trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batchSize, shuffle=True)
valLoader = torch.utils.data.DataLoader(val_data, batch_size=batchSize, shuffle=True)

dataLoaders = {
    'train': trainLoader,
    'val': valLoader
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# pretrained se-resnet50 on ImageNet
se_resnet_hub_model = torch.hub.load(
    'moskomule/senet.pytorch',
    'se_resnet50',
    pretrained=True,)

se_resnet_base_layers = [] 

# traverse each element of hub model and add it to a list
for name, m in se_resnet_hub_model.named_children():
    se_resnet_base_layers.append(m)  

# two class mask and marker
numClass = 2
model = USENet(numClass, se_resnet_base_layers).to(device)

# summarize the network
summary(model, [(3, 992, 992)])

# using Adam optimizer with learning rate 1e-4
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# decrease learning rate by 0.1 after each 30th epoch
lrScheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# train model with 100 epoch
trainModel(dataLoaders, model, optimizer, lrScheduler, numEpochs=100, modelN = modelName)
