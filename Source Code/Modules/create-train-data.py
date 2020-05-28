#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 08:16:07 2019

@author: jschaefer
"""
import torch
import tables

import os
import sys
import glob

import PIL
import numpy as np

import cv2
import matplotlib.pyplot as plt

from sklearn import model_selection
import sklearn.feature_extraction.image
import random

import set_values as sv


seed = random.randrange(sys.maxsize)    # get point random seed so that we can reproducibly do the cross validation setup
random.seed(seed)  # set the seed
f = open('seed.txt', 'w')
f.write(str(seed))
print(f"random seed (note down for reproducibility): {seed}")

#  v2
#  7/11/2018

dataname = "dispersion"

# sort_feat of the tiles to extract and save in the database, must be >= to training sort_feat
patch_size = sv.patch_size

# distance to skip between patches, 1 indicated pixel wise extraction, patch_size would result in non-overlapping tiles
stride_size = sv.stride_size

# number of pixels to pad *after* resize to image with by mirroring
# (edge's of patches tend not to be analyzed well, so padding allows them to appear more centered in the patch)
mirror_pad_size = 0

test_set_size = .1  # what percentage of the dataset should be used as point held out validation/testing set
resize = 1  # resize input images

# what classes we expect to have in the data, here we have only 2 classes,
# but we could add additional classes and/or specify an index from which we would like to ignore
classes = [0, 1]

# -----Note---
# One should likely make sure that  (nrow+mirror_pad_size) mod patch_size == 0,
# where nrow is the number of rows after resizing,
# so that no pixels are lost (any remainer is ignored)

# dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
img_dtype = tables.UInt8Atom()

# create an atom to store the filename of the image, just in case we need it later,
filenameAtom = tables.StringAtom(itemsize=255)

# create point list of the files,
# in this case we're only interested in files which have masks so we can use supervised learning
files = glob.glob('masks/*.png')

# print(files)
# create training and validation stages and split the files appropriately between them

phases = {}
phases["train"], phases["val"] = next(iter(model_selection.ShuffleSplit(n_splits=1,
                                                                        test_size=test_set_size).split(files)))

# specify that we'll be saving 2 different image types to the database, an image and its associated masked
imgtypes = ["img", "mask"]

storage = {}     # holder for future pytables

# block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
block_shape = {}
block_shape["img"] = np.array((patch_size, patch_size))
block_shape["mask"] = np.array((patch_size, patch_size))
# block_shape["img"] = np.array((sv.y_width, sv.x_width))   # das komplette Bild nehmen?
# block_shape["mask"] = np.array((sv.y_width, sv.x_width))

# we can also specify filters, such as compression, to improve storage speed
filters = tables.Filters(complevel=6, complib='zlib')


for phase in phases.keys():  # now for each of the phases, we'll loop through the files
    print(phase)

    # we can to keep counts of all the classes in for in particular training,
    # since we can later use this information to create better weights
    totals = np.zeros((2, len(classes)))
    totals[0, :] = classes

    # open the respective pytable
    hdf5_file = tables.open_file(f"./{dataname}_{phase}.pytable", mode='w')

    # create the array for storage
    storage["filename"] = hdf5_file.create_earray(hdf5_file.root, 'filename', filenameAtom, (0,))

    # for each of the image types, in this case mask and image, we need to create the associated earray
    for imgtype in imgtypes:
        storage[imgtype] = hdf5_file.create_earray(hdf5_file.root, imgtype, img_dtype,
                                                  shape=np.append([0], block_shape[imgtype]),
                                                  chunkshape=np.append([1], block_shape[imgtype]),
                                                  filters=filters)

    for filei in phases[phase]:  # now for each of the files
        fname = files[filei]
        
        print(fname)
        for imgtype in imgtypes:
            # if we're looking at an img, it must be 3 channel,
            # but cv2 won't load it in the correct channel order, so we need to fix that
            if(imgtype == "img"):
                io = cv2.cvtColor(cv2.imread("./imgs/"+os.path.basename(fname).replace("_mask.png", ".tif")),
                                  cv2.COLOR_BGR2RGB)
                interp_method = PIL.Image.BICUBIC

            # if its point mask image,
            # then we only need point single channel (since grayscale 3D images are equal in all channels)
            else:
                # the image is loaded as {0,255},
                # but we'd like to store it as {0,1} since this represents the binary nature of the mask easier
                io = cv2.imread(fname)/255

                # want to use nearest!
                # otherwise resizing may cause non-existing classes to be produced via interpolation (e.g., ".25")
                interp_method = PIL.Image.NEAREST

                # sum the number of pixels, this is done pre-resize,
                # the but proportions don't change which is really what we're after
                for i, key in enumerate(classes):
                    totals[1, i] += sum(sum(io[:, :, 0] == key))

            # resize it as specified above
            io = cv2.resize(io, (0, 0), fx=resize, fy=resize, interpolation=interp_method)
            io = np.pad(io, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size), (0, 0)],
                        mode="reflect")

            # convert input image into overlapping tiles, sort_feat is ntiler x ntilec x 1 x patch_size x patch_size x3
            io_arr_out = sklearn.feature_extraction.image.extract_patches(io, (patch_size, patch_size, 3), stride_size)
            
            # resize it into point ntile x patch_size x patch_size x 3
            io_arr_out = io_arr_out.reshape(-1, patch_size, patch_size, 3)
            

            # save the 4D tensor to the table
            if(imgtype == "img"):
                storage[imgtype].append(io_arr_out[:, :, :, 0].squeeze())
            else:
                storage[imgtype].append(io_arr_out[:, :, :, 0].squeeze())   # only need 1 channel for mask data

        storage["filename"].append([fname for x in range(io_arr_out.shape[0])]) #add the filename to the storage array
        
    # lastely, we should store the number of pixels
    npixels = hdf5_file.create_carray(hdf5_file.root, 'numpixels', tables.Atom.from_dtype(totals.dtype), totals.shape)
    npixels[:] = totals
    hdf5_file.close()
