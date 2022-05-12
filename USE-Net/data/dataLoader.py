import torch
import numpy as np
import torchvision.transforms as transforms

from torch.utils.data.dataset import Dataset
from skimage.transform import resize
from PIL import Image

# path loader
class USENetPathLoader(Dataset):
    def __init__(self, image_paths,  target_paths, marker_paths):

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.marker_paths = marker_paths

    def __getitem__(self, index):

        x = self.image_paths[index]
        y = self.target_paths[index]
        z = self.marker_paths[index]

        return x, y, z

    def __len__(self):

        return len(self.image_paths)

# train data loader
class USENetDataLoader(Dataset):
    def __init__(self, all_paths):

        self.all_paths = all_paths

        self.transforms = transforms.Compose([
            transforms.Resize((992,992)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        imgPath, labelPath, markerPath = self.all_paths[index]

        image = Image.open(imgPath)
        x = self.transforms(image)

        mask = Image.open(labelPath)
        marker = Image.open(markerPath)

        y = np.array(mask)
        y = resize(y, (992, 992))

        m = np.array(marker)
        m = resize(m, (992, 992))

        yNew = np.zeros([2, 992, 992])
        yNew[0,:,:] = y
        yNew[1,:,:] = m

        yNew = torch.from_numpy(yNew).long()

        return x, yNew

    def __len__(self):

        return len(self.all_paths)

# test data loader
class USENetTestDataLoader(Dataset):
    def __init__(self, image_paths):

        self.image_paths = image_paths

        self.transforms = transforms.Compose([
            transforms.Resize((992,992)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        x = self.transforms(image)

        return x


    def __len__(self):

        return len(self.image_paths)