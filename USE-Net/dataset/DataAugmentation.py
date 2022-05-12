import numpy as np
from skimage.io import imshow, imread
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.grid'] = False
mpl.rcParams['image.interpolation'] = 'nearest'
mpl.rcParams['figure.figsize'] = 15, 10

def show(img):
    npimg = img.numpy()
    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

import os
import random

import torch
import torch.utils.data as data

from PIL import Image

from imgaug import augmenters as iaa
import imgaug as ia

class SegmentationDatasetImgaug(data.Dataset):
    IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
    
    @staticmethod
    def _load_input_image(path):
        return imread(path)
    
    @staticmethod
    def _load_target_image(path):
        return imread(path, as_gray=True)[..., np.newaxis]
            
    def __init__(self, input_root, input_root2, target_root, target_root2, transform=None, input_only=None):
        self.input_root = input_root
        self.input_root2 = input_root2
        self.target_root = target_root
        self.target_root2 = target_root2
        self.transform = transform
        self.input_only = input_only
                
        self.input_ids = sorted(img for img in os.listdir(self.input_root)
                                if self._isimage(img, self.IMG_EXTENSIONS))
        
        self.input_ids2 = sorted(img for img in os.listdir(self.input_root2)
                                if self._isimage(img, self.IMG_EXTENSIONS))
        
        self.target_ids = sorted(img for img in os.listdir(self.target_root)
                                 if self._isimage(img, self.IMG_EXTENSIONS))
        
        self.target_ids2 = sorted(img for img in os.listdir(self.target_root2)
                                 if self._isimage(img, self.IMG_EXTENSIONS))
        
        assert(len(self.input_ids) == len(self.target_ids))
        assert(len(self.input_ids2) == len(self.target_ids2))
        
    def _activator_masks(self, images, augmenter, parents, default):
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default
    
    def __getitem__(self, idx):
        input_img = self._load_input_image(
            os.path.join(self.input_root, self.input_ids[idx]))
        input_img2 = self._load_input_image(
            os.path.join(self.input_root2, self.input_ids2[idx]))
        target_img = self._load_target_image(
            os.path.join(self.target_root, self.target_ids[idx]))
        target_img2 = self._load_target_image(
            os.path.join(self.target_root2, self.target_ids2[idx]))
        
        if self.transform:
            det_tf = self.transform.to_deterministic()
            input_img = det_tf.augment_image(input_img)
            input_img2 = det_tf.augment_image(input_img2)
            target_img = det_tf.augment_image(
                target_img,
                hooks=ia.HooksImages(activator=self._activator_masks))
            target_img2 = det_tf.augment_image(
                target_img2,
                hooks=ia.HooksImages(activator=self._activator_masks))
            
        to_tensor = transforms.ToTensor()
        input_img = to_tensor(input_img)
        input_img2 = to_tensor(input_img2)
        target_img = to_tensor(target_img)
        target_img2 = to_tensor(target_img2)
            
        return input_img, input_img2, target_img, target_img2, self.input_ids[idx]
        
    def __len__(self):
        return len(self.input_ids)


augs = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-25, 25),
               translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, mode='constant')
])

# train data path that needs to be augmented
ds3 = SegmentationDatasetImgaug(
    './trainData/Inputs/', './trainData/SPCN/', './trainData/BinaryMask/', './trainData/Marker/',
    transform=augs
)

import glob

folderData = sorted(glob.glob("./trainData/Inputs/*.png"))


import imageio

# path where augmented data will be stored
inputDir = os.path.join('./augTrainData/Inputs/')
inputDir2 = os.path.join('./augTrainData/SPCN/')

labelDir = os.path.join('./augTrainData/BinaryMask/')
labelDir2 = os.path.join('./augTrainData/Marker/')

for i in range(149):
    
    count = 0;
    
    print('Augmentation step: ')
    print(i)
    
    for inputs, inputs2, labels, labels2, selfInput in ds3:
        
        label = labels.numpy()
        outPred = label[0].squeeze()
        outPredNorm = 255 * outPred
        outPredUint8 = outPredNorm.astype(np.uint8)
        
        label2 = labels2.numpy()
        outPred2 = label2[0].squeeze()
        outPredNorm2 = 255 * outPred2
        outPredUint8_2 = outPredNorm2.astype(np.uint8)
       
        inputA = inputs.numpy()
        finaInput = np.transpose(inputA, (1, 2, 0))
        finaInput = 255 * finaInput
        inputFinal = finaInput.astype(np.uint8)
        
        inputB = inputs2.numpy()
        finaInputB = np.transpose(inputB, (1, 2, 0))
        finaInputB = 255 * finaInputB
        inputFinalB = finaInputB.astype(np.uint8)
        
        fname = os.path.basename(folderData[count]).replace('.png','')
        fname = fname + '_aug_' + str(i)+'.png'

        imageio.imwrite(labelDir + fname, outPredUint8)
        imageio.imwrite(labelDir2 + fname, outPredUint8_2)
        
        imageio.imwrite(inputDir + fname, inputFinal)
        imageio.imwrite(inputDir2 + fname, inputFinalB)

        count = count + 1

