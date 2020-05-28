# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:02:55 2019

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
from skimage.morphology import skeletonize
from skimage.feature import peak_local_max
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from unet import UNet
from scipy import misc
from scipy import ndimage
from matplotlib import pyplot as plt

import skimage

import csv
import PIL


import main_methods
import set_values as sv


def plot_kernels(tensor, l, n, num_cols=4, cmap="gray"):
    if not len(tensor.shape) == 4:
        raise Exception("assumes a 4D tensor")
    #    if not tensor.shape[1]==3:
    #        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0] * tensor.shape[1]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure()
    i = 0
    t = tensor.data.numpy()
    c = 0
    for t1 in t:
        for t2 in t1:
            # width=int(t2.shape[1]*100)
            # height=int(t2.shape[0]*100)
            # dim=(width,height)
            i += 1
            # ax1 = fig.add_subplot(num_rows,num_cols,i)
            # plt.imshow(cv2.resize(t2,dim,interpolation=cv2.INTER_AREA) , cmap=cmap)
            plt.imshow(t2, cmap=cmap)
            plt.axis('off')
            # print(i)
            # plt.set_xticklabels([])
            # plt.set_yticklabels([])
            plt.savefig("U" + str(l) + "/" + "L" + str(n) + "/" + str(c) + '.jpg', bbox_inches='tight')
            c += 1
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()


def plot_kernels_down(tensor, l, n, num_cols=4, cmap="gray"):
    if not len(tensor.shape) == 4:
        raise Exception("assumes a 4D tensor")
    #    if not tensor.shape[1]==3:
    #        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0] * tensor.shape[1]
    num_rows = 1 + num_kernels // num_cols
    fig = plt.figure()
    i = 0
    t = tensor.data.numpy()
    c = 0
    for t1 in t:
        for t2 in t1:
            # width=int(t2.shape[1]*100)
            # height=int(t2.shape[0]*100)
            # dim=(width,height)
            i += 1
            # ax1 = fig.add_subplot(num_rows,num_cols,i)
            # plt.imshow(cv2.resize(t2,dim,interpolation=cv2.INTER_AREA) , cmap=cmap)
            plt.imshow(t2, cmap=cmap)
            plt.axis('off')
            # print(i)
            # plt.set_xticklabels([])
            # plt.set_yticklabels([])
            plt.savefig("D" + str(l) + "/" + "N" + str(n) + "/" + str(c) + '.jpg', bbox_inches='tight')
            c += 1
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # plt.show()


class LayerActivations():
    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


# v2
# 28/11/2018

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
        img = self.img[index, :, :]  # ,:
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


dataname = "dispersion"
ignore_index = -100  # Unet has the possibility of masking out pixels in the output image, we can specify the index value here (though not used)
gpuid = 0

# csv=open('dia.csv','w')
patch_size = sv.patch_size  # 192#220 #should match the value used to train the network
batch_size = 1  # nicer to have a single batch so that we can iterately view the output, while not consuming too much
edge_weight = sv.edge_weight

# Cpu or gpu support-----------------------------------------------------
# print(torch.cuda.get_device_properties(gpuid))
# torch.cuda.set_device(gpuid)
# device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
device = ('cpu')

checkpoint = torch.load(f"{dataname}_unet_best_model.pth", map_location='cpu')

# load the model, note that the paramters are coming from the checkpoint, since the architecture of the model needs to exactly match the weights saved
model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"], padding=checkpoint["padding"],
             depth=checkpoint["depth"],
             wf=checkpoint["wf"], up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
model.load_state_dict(checkpoint["model_dict"])
# note that since we need the transofrmations to be reproducible for both masks and images
# we do the spatial transformations first, and afterwards do any color augmentations

# in the case of using this for output generation, we want to use the original images since they will give a better sense of the exepected
# output when used on the rest of the dataset, as a result, we disable all unnecessary augmentation.
# the only component that remains here is the randomcrop, to ensure that regardless of the size of the image
# in the database, we extract an appropriately sized patch
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # these need to be in a reproducible order, first affine transforms and then color
    # transforms.RandomResizedCrop(size=patch_size),
    # transforms.RandomRotation(180),
    # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
    # transforms.RandomGrayscale(),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    # these need to be in a reproducible order, first affine transforms and then color
    # transforms.RandomResizedCrop(size=patch_size,interpolation=PIL.Image.NEAREST),
    # transforms.RandomRotation(180),
])

# set the model to evaluation mode, since we're only generating output and not doing any back propogation
model.eval()


path = 'imgs/'
number_of_images = len(os.listdir(path))


for image_number, image_name in enumerate(os.listdir(path)):

    file_name = "imgs/" + image_name
    loaded_image = cv2.imread(file_name, 0) / 255  

    # split the image, because the whole image doesn't fit in the cnn format (it has to be quadratic)!
    cropped_image = main_methods.crop_image(loaded_image)
    cropped_image = cropped_image.to(device)  # [NBATCH, 3, H, W]

    for i in range(5):  # Tiefe Netz
        # if i >3:
        try:
            os.mkdir("D" + str(i))
        except:
            print(" ")
        for j in range(6):  # Filter
            try:
                os.mkdir("D" + str(i) + "/" + "N" + str(j))
            except:
                print(" ")
            dr = LayerActivations(model.down_path[i].block[j])
            # dr=LayerActivations(model.up_path[i])
            print("i: " + str(i))
            print("j: " + str(j))

            output = model(cropped_image.to(device))

            plot_kernels_down(dr.features, i, j, 12, cmap='gray')

    for i in range(5):
        # if i >3:
        try:
            os.mkdir("U" + str(i))
        except:
            print(" ")
        for j in range(6):
            try:
                os.mkdir("U" + str(i) + "/" + "L" + str(j))
            except:
                print(" ")
            # dr=LayerActivations(model.down_path[i].block[j])
            dr = LayerActivations(model.up_path[i])
            print("i: " + str(i))
            print("j: " + str(j))

            output = model(cropped_image.to(device))

            plot_kernels(dr.features, i, j, 12, cmap='gray')
