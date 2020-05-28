#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:16:07 2019

@author: jschaefer
@editor: sliebrecht
"""
import random, sys
import cv2
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage
import skimage
import time

import tables
from skimage import io, morphology
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet

import PIL

import set_values as sv


# this defines our dataset class which will be used by the dataloader
class Dataset(object):
    def __init__(self, fname, img_transform=None, mask_transform=None, edge_weight=False):
        # nothing special here, just internalizing the constructor parameters
        self.fname = fname
        self.edge_weight = edge_weight

        self.img_transform = img_transform
        self.mask_transform = mask_transform

        self.tables = tables.open_file(self.fname)
        self.numpixels = self.tables.root.numpixels[:]
        self.nitems = self.tables.root.img.shape[0]
        self.tables.close()

        self.img = None
        self.mask = None

    def __getitem__(self, index):
        # opening should be done in __init__ but seems to be
        # an issue with multithreading so doing here
        if (self.img is None):  # open in thread
            self.tables = tables.open_file(self.fname)
            self.img = self.tables.root.img
            self.mask = self.tables.root.mask

        # get the requested image and mask from the pytable
        img = self.img[index, :, :]
        mask = self.mask[index, :, :]

        # the original Unet paper assignes increased weights to the edges of the annotated objects
        # their method is more sophistocated, but this one is faster, we simply dilate the mask and
        # highlight all the pixels which were "added"
        if (self.edge_weight):
            weight = scipy.ndimage.morphology.binary_dilation(mask == 1, iterations=2) & ~mask
        else:  # otherwise the edge weight is all ones and thus has no affect
            weight = np.ones(mask.shape, dtype=mask.dtype)

        mask = mask[:, :, None].repeat(3, axis=2)  # in order to use the transformations given by torchvision
        weight = weight[:, :, None].repeat(3,
                                           axis=2)  # inputs need to be 3D, so here we convert from 1d to 3d by repetition

        img_new = img
        mask_new = mask
        weight_new = weight

        seed = random.randrange(sys.maxsize)  # get a random seed so that we can reproducibly do the transofrmations
        if self.img_transform is not None:
            random.seed(seed)  # apply this seed to img transforms
            img_new = self.img_transform(img)

        if self.mask_transform is not None:
            random.seed(seed)
            mask_new = self.mask_transform(mask)
            mask_new = np.asarray(mask_new)[:, :, 0].squeeze()

            random.seed(seed)
            weight_new = self.mask_transform(weight)
            weight_new = np.asarray(weight_new)[:, :, 0].squeeze()

        return img_new, mask_new, weight_new

    def __len__(self):
        return self.nitems


dataname = "dispersion"  # should match the value used to train the network, will be used to load the appropirate model
gpuid = 0

patch_size = sv.patch_size  # should match the value used to train the network
batch_size = sv.batch_size  # nicer to have a single batch so that we can iterately view the output, while not consuming too much
edge_weight = sv.edge_weight

# Cpu or gpu support-----------------------------------------------------
# print(torch.cuda.get_device_properties(gpuid))
# torch.cuda.set_device(gpuid)
# device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
device = ('cpu')

checkpoint = torch.load(f"{dataname}_unet_best_model.pth", map_location='cpu')

# load the model, note that the paramters are coming from the checkpoint,
# since the architecture of the model needs to exactly match the weights saved
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"], padding=checkpoint["padding"],
             depth=checkpoint["depth"], wf=checkpoint["wf"], up_mode=checkpoint["up_mode"],
             batch_norm=checkpoint["batch_norm"]).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
model.load_state_dict(checkpoint["model_dict"])

# note that since we need the transofrmations to be reproducible for both masks and images
# we do the spatial transformations first, and afterwards do any color augmentations

# in the case of using this for output generation,
# we want to use the original images since they will give a better sense of the exepected
# output when used on the rest of the dataset, as a result, we disable all unnecessary augmentation.
# the only component that remains here is the randomcrop, to ensure that regardless of the sort_feat of the image
# in the database, we extract an appropriately sized patch
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),

    # these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # transforms.RandomResizedCrop(sort_feat=patch_size),
    # transforms.RandomRotation(180),
    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
    # transforms.RandomGrayscale(),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),

    # these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # transforms.RandomResizedCrop(sort_feat=patch_size,interpolation=PIL.Image.NEAREST),
    # transforms.RandomRotation(180),
])

phases = ["val"]
dataset = {}
dataLoader = {}
for phase in phases:
    print(phase)
    dataset[phase] = Dataset(f"./{dataname}_{phase}.pytable", img_transform=img_transform,
                             mask_transform=mask_transform, edge_weight=edge_weight)
    dataLoader[phase] = DataLoader(dataset[phase], batch_size=1, shuffle=True, num_workers=0,
                                   pin_memory=True)  # ,pin_memory=True)

# %matplotlib inline

# set the model to evaluation mode, since we're only generating output and not doing any back propogation
model.eval()

for ii, (X, y, y_weight) in enumerate(dataLoader["val"]):


    # print(len(dataLoader["val"]))

    X = X.to(device)  # [NBATCH, 3, H, W]

    y = y.type('torch.LongTensor').to(device)  # [NBATCH, H, W] with class indices (0, 1)

    output = model(X)  # [NBATCH, 2, H, W]

    output = output.detach().squeeze().cpu().numpy()  # get output and pull it to CPU
    output = np.moveaxis(output, 0, -1)  # reshape moving last dimension
    for k in range(batch_size):
        # s=k # delete?
        # output[10:100, :, 1] # from pixel 10:100 and all y-Pixels!
        bound = cv2.normalize(output[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite('netz_bilder/' + str(ii) + ".png", bound)

        X = X.squeeze()

        # ax = None
        # if ax is None:
        #     fig, ax = plt.subplots()
        # ax.imshow(cropped_image)
        # plt.show()


        # adding two images, 2 ways:
        # 1. as Numpy-Operation:
        #   res = img1 + img2   # this works with modulo
        # 2. cv2.add()  # but theres a diffrence between those two! This is better!?! look opencvtutorials!

        # ax[0].imshow(output[:, :, 1], cmap="gray")
        # ax[1].imshow((output[:, :, 0]), cmap="gray")
        # ax[2].imshow(np.moveaxis(y.detach().squeeze().cpu().numpy(), 0, -1) * 255, cmap="gray")
        # ax[3].imshow(np.moveaxis(cropped_image.detach().squeeze().cpu().numpy(), 0, -1), cmap="gray")
        # plt.pause(1)
