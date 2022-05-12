import torch
import numpy as np
import torchvision
import glob
import os
import imageio
import cv2
import torch.nn.functional as F
import time
import torch.hub

from datetime import timedelta
from data.dataLoader import USENetTestDataLoader
from nets.USENet import USENet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# model name
modelName = 'USENet_Final'

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

# load trained model
model.load_state_dict(torch.load('./models/' + modelName + '.pt'))
model.eval() 

# start timer 
startTime = time.time()

# path to the test data
folderData = sorted(glob.glob("./dataset/testData/SPCN/*.png"))
print(len(folderData))

# load test data
testDataset = USENetTestDataLoader(folderData)
print(len(testDataset))

testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False)

# set mask path
maskDir = os.path.join('./output/' + modelName + '/Mask/')
# create path if not exist
if not os.path.exists(maskDir):
    os.makedirs(maskDir)

# set marker path
markerDir = os.path.join('./output/' + modelName + '/Marker/')
# create path if not exist
if not os.path.exists(markerDir):
    os.makedirs(markerDir)


for i, (inputs) in enumerate(testLoader):
        
    inputs = inputs.to(device)   
    inputs = inputs.float()

    # Predict
    pred = model(inputs)
        
    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    
    outPred = pred[0].squeeze()

    outPredMask = cv2.resize(outPred[0,:,:], dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)
    outPredMarker = cv2.resize(outPred[1,:,:], dsize=(1000, 1000), interpolation=cv2.INTER_LINEAR)

    outPredMaskNorm = 255 * outPredMask
    outPredMaskUint8 = outPredMaskNorm.astype(np.uint8)

    outPredMarkerNorm = 255 * outPredMarker
    outPredMarkerUint8 = outPredMarkerNorm.astype(np.uint8)
        
    fname = os.path.basename(folderData[i])

    print(maskDir + fname)
        
    imageio.imwrite(maskDir + fname, outPredMaskUint8)
    imageio.imwrite(markerDir + fname, outPredMarkerUint8)

finalTime = time.time() - startTime
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(finalTime))
print(msg)




