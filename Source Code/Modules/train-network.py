#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:21:17 2019

@author: jschaefer
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from unet import UNet    # code borrowed from https://github.com/jvanvugt/pytorch-unet

import PIL
import matplotlib.pyplot as plt
import cv2

import numpy as np
import sys, glob

from tensorboardX import SummaryWriter

import scipy.ndimage

import time
import math
import tables

import random

from sklearn.metrics import confusion_matrix

import set_values as sv


# helper function for pretty printing of current time and remaining time
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


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
        with tables.open_file(self.fname, 'r') as db:
            self.img = db.root.img
            self.mask = db.root.mask

            # get the requested image and mask from the pytable
            img = self.img[index, :, :]
            mask = self.mask[index, :, :]

        # the original Unet paper assignes increased weights to the edges of the annotated objects
        # their method is more sophistocated, but this one is faster, we simply dilate the mask and
        # highlight all the pixels which were "added"
        if self.edge_weight:
            weight = scipy.ndimage.morphology.binary_dilation(mask == 1, iterations=2) & ~mask
        else:   # otherwise the edge weight is all ones and thus has no affect
            weight = np.ones(mask.shape, dtype=mask.dtype)

        # in order to use the transformations given by torchvision
        mask = mask[:, :, None].repeat(3, axis=2)

        # inputs need to be 3D, so here we convert from 1d to 3d by repetition
        weight = weight[:, :, None].repeat(3, axis=2)

        img_new = img
        mask_new = mask
        weight_new = weight

        # get a random seed so that we can reproducibly do the transformations
        seed = random.randrange(sys.maxsize)
        if self.img_transform is not None:
            random.seed(seed)   # apply this seed to img transforms
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


class LayerActivations():

    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output.cpu()

    def remove(self):
        self.hook.remove()


dataname = "dispersion"

# Unet has the possibility of masking out pixels in the output image,
# we can specify the index value here (though not used)
ignore_index = -100
gpuid = 0

# --- unet params
# these parameters get fed directly into the UNET class, and more description of them can be discovered there
n_classes = 2   # number of classes in the data mask that we'll aim to predict
in_channels = 1  # input channel of the data, RGB = 3
padding = True   # should levels be padded
depth = sv.depth   # 6       # depth of the network
wf = sv.wf           # wf (int): number of filters in the first layer is 2**wf, was 6
up_mode = 'upconv'  # should we simply upsample the mask, or should we try and learn an interpolation
batch_norm = True    # should we use batch normalization between the layers

# --- training params
patch_size = sv.patch_size   # 256 or multi of it?
batch_size = sv.batch_size
num_epochs = sv.num_epochs

# edges tend to be the most poorly segmented given how little area they occupy in the training set,
# this paramter boosts their values along the lines of the original UNET paper
edge_weight = sv.edge_weight
phases = ["train", "val"]    # how many phases did we create databases for?

# when should we do valiation? note that validation is time consuming,
# so as opposed to doing for both training and validation, we do it only for vlaidation at the end of the epoch
validation_phases = ["val"]

# specify if we should use a GPU (cuda) or only the CPU
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_properties(gpuid))
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_properties(gpuid))
# torch.cuda.set_device(gpuid)
# device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# build the model according to the paramters specified above and copy it to the GPU.
# finally print out the number of trainable parameters
model = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding, depth=depth, wf=wf, up_mode=up_mode,
             batch_norm=batch_norm).to(device)
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

# note that since we need the transofrmations to be reproducible for both masks and images
# we do the spatial transformations first, and afterwards do any color augmentations
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),

    # these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    transforms.RandomResizedCrop(size=patch_size),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
    transforms.RandomGrayscale(),
    transforms.ToTensor()
    ])


mask_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),

    # these need to be in a reproducible order, first affine transforms and then color
    transforms.RandomCrop(size=(patch_size, patch_size), pad_if_needed=True),
    transforms.RandomResizedCrop(size=patch_size, interpolation=PIL.Image.NEAREST),
    transforms.RandomRotation(180),
    ])


dataset = {}
dataLoader = {}

# now for each of the phases, we're creating the dataloader interestingly, given the batch sort_feat,
# i've not seen any improvements from using a num_workers>0
for phase in phases:

    print(phase)
    dataset[phase] = Dataset(f"./{dataname}_{phase}.pytable", img_transform=img_transform,
                             mask_transform=mask_transform, edge_weight=edge_weight)
    dataLoader[phase] = DataLoader(dataset[phase], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# visualize a single example to verify that it is correct
# (img,patch_mask,patch_mask_weight)=dataset["train"][10]
# fig, ax = plt.subplots(1,4, figsize=(10,4))  # 1 row, 2 columns

# build output showing original patch  (after augmentation), class = 1 mask,
# weighting mask, overall mask (to see any ignored classes)
# ax[0].imshow(np.moveaxis(img.numpy(),0,-1))
# ax[1].imshow(patch_mask==1)
# ax[2].imshow(patch_mask_weight)
# ax[3].imshow(patch_mask)

# adam is going to be the most robust, though perhaps not the best performing, typically a good place to start
optim = torch.optim.Adam(model.parameters())
# optim = torch.optim.SGD(model.parameters(), lr=.1, momentum=0.9, weight_decay=0.0005)

# we have the ability to weight individual classes, in this case we'll do so based on their presense in the trainingset
# to avoid biasing any particular class
nclasses = dataset["train"].numpixels.shape[1]
class_weight = dataset["train"].numpixels[1, 0:2]     # don't take ignored class into account here
class_weight = torch.from_numpy(1-class_weight/class_weight.sum()).type('torch.FloatTensor').to(device)

# show final used weights, make sure that they're reasonable before continouing
print(class_weight)

# reduce = False makes sure we get a 2D output instead of a 1D "su
criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=ignore_index, reduce=False)


# empties the output_file
with open('new_output.txt', 'w') as output_file:
    pass

writer = SummaryWriter()    # open the tensorboard visualiser
best_loss_on_test = np.Infinity
edge_weight = torch.tensor(edge_weight).to(device)
start_time = time.time()
for epoch in range(num_epochs):
    # zero out epoch based performance variables
    all_acc = {key: 0 for key in phases}
    all_loss = {key: torch.zeros(0).to(device) for key in phases}
    cmatrix = {key: np.zeros((2, 2)) for key in phases}

    for phase in phases:    # iterate through both training and validation states

        if phase == 'train':
            model.train()  # Set model to training mode
        else:    # when in eval mode, we don't want parameters to be updated
            model.eval()   # Set model to evaluate mode

        for ii, (X, y, y_weight) in enumerate(dataLoader[phase]):   # for each of the batches
            X = X.to(device)  # [Nbatch, 3, H, W]
            y_weight = y_weight.type('torch.FloatTensor').to(device)
            y = y.type('torch.LongTensor').to(device)  # [Nbatch, H, W] with class indices (0, 1)

            # dynamically set gradient computation, in case of validation, this isn't needed
            # disabling is good practice and improves inference time
            with torch.set_grad_enabled(phase == 'train'):

                prediction = model(X)  # [N, Nclass, H, W]
                loss_matrix = criterion(prediction, y)
                loss = (loss_matrix * (edge_weight**y_weight)).mean()   # can skip if edge weight==1

                if phase == "train":  # in case we're in train mode, need to do back propogation
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    train_loss = loss

                all_loss[phase] = torch.cat((all_loss[phase], loss.detach().view(1, -1)))

                if phase in validation_phases:  # if this phase is part of validation, compute confusion matrix
                    p = prediction[:, :, :, :].detach().cpu().numpy()
                    cpredflat = np.argmax(p, axis=1).flatten()
                    yflat = y.cpu().numpy().flatten()

                    cmatrix[phase] = cmatrix[phase]+confusion_matrix(yflat, cpredflat, labels=range(n_classes))

        all_acc[phase] = (cmatrix[phase]/cmatrix[phase].sum()).trace()
        all_loss[phase] = all_loss[phase].cpu().numpy().mean()

        # save metrics to tensorboard
        writer.add_scalar(f'{phase}/loss', all_loss[phase], epoch)
        if phase in validation_phases:
            writer.add_scalar(f'{phase}/acc', all_acc[phase], epoch)
            writer.add_scalar(f'{phase}/TN', cmatrix[phase][0, 0], epoch)
            writer.add_scalar(f'{phase}/TP', cmatrix[phase][1, 1], epoch)
            writer.add_scalar(f'{phase}/FP', cmatrix[phase][0, 1], epoch)
            writer.add_scalar(f'{phase}/FN', cmatrix[phase][1, 0], epoch)
            writer.add_scalar(f'{phase}/TNR', cmatrix[phase][0, 0]/(cmatrix[phase][0, 0]+cmatrix[phase][0, 1]), epoch)
            writer.add_scalar(f'{phase}/TPR', cmatrix[phase][1, 1]/(cmatrix[phase][1, 1]+cmatrix[phase][1, 0]), epoch)

    print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs),
                                                 epoch+1, num_epochs, (epoch+1) / num_epochs * 100, all_loss["train"],
                                                                   all_loss["val"]), end="")

    with open('new_output.txt', 'a') as output_file:
        print('%s ([%d/%d] %d%%), train loss: %.4f test loss: %.4f' % (timeSince(start_time, (epoch+1) / num_epochs),
                                                 epoch+1, num_epochs, (epoch+1) / num_epochs * 100, all_loss["train"],
                                                                   all_loss["val"]), end="", file=output_file)
        if all_loss["val"] < best_loss_on_test:
            best_loss_on_test = all_loss["val"]
            print("  **\n", file=output_file)
        else:
            print("\n", file=output_file)

    # if current loss is the best we've seen, save model state with all variables
    # necessary for recreation
    if all_loss["val"] < best_loss_on_test:
        best_loss_on_test = all_loss["val"]
        print("  **")
        state = {'epoch': epoch + 1,
                 'model_dict': model.state_dict(),
                 'optim_dict': optim.state_dict(),
                 'best_loss_on_test': all_loss,
                 'n_classes': n_classes,
                 'in_channels': in_channels,
                 'padding': padding,
                 'depth': depth,
                 'wf': wf,
                 'up_mode': up_mode, 'batch_norm': batch_norm}

        torch.save(state, f"{dataname}_unet_best_model.pth")
    else:
        print("")

# ----- generate output
# load best model
checkpoint = torch.load(f"{dataname}_unet_best_model.pth")
model.load_state_dict(checkpoint["model_dict"])
[img, mask, mask_weight] = dataset["val"][5]
output = model(img[None, ::].to(device))
output = output.detach().squeeze().cpu().numpy()
output = np.moveaxis(output, 0, -1)
output.shape

