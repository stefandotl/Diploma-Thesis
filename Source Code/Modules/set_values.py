import random

"""
This module links all comprehensive variables in one file
"""


############################
# For train_network.py, test-result.py and create-train-data

# sort_feat of the tiles to extract and save in the database, must be >= to training sort_feat
patch_size = 512  # 256 or multi of it?

############################
# For train_network.py, test-result.py

# or batch_size=1? nicer to have a single batch so that we can iterately view the output, while not consuming too much?
batch_size = 3

# edges tend to be the most poorly segmented given how little area they occupy in the training set,
# this paramter boosts their values along the lines of the original UNET paper
edge_weight = 8.0  # or 1.0?

############################
# Only train_network.py

depth = 5   # 6       # depth of the network
wf = 3           # wf (int): number of filters in the first layer is 2**wf, was 6

############################
# Only create_train_data.py

x_width = 1280  # Bildbreite (Basler Kamera)
y_width = 1024  # Bildhöhe

# distance to skip between patches, 1 indicated pixel wise extraction, patch_size would result in non-overlapping tiles
stride_size = 256
num_epochs = 100


# ############################
# Only create_image_new.py

num_imgs = 2 # number_of_images to create (1000 recommended)
max_cover = 0.60  # how much percent can an ellipse cover other ellipses (example: 0.3 = 30 %)
alphaT = round(random.uniform(0.1, 0.15), 3)  # maximaler Phasenanteil!
#alphaT = round(random.uniform(0.05, 0.1), 3)  # maximaler Phasenanteil!


###########################
# detection_methods both versions!

min_area = 6000     # 6000 for ed_v1 recommended, 150 for ed_2?

# filter_points
dist = 10   # 10 for ed_v1? how far points can be away from each other without being considered as extreme Points
dist_fil = 5    # how far a Point shall minimum being away from an filtered Point without get filtered aswell.

# filtter_shape
# how far a point from an contour can be away from the corresponding fitted ellipse without being discarded
max_dist = 1    # for artifical ellipse: max_dist = 1, for real images max_dist = 3 recommended!

# hull_needed
ratio = 0.95    # recommended 0.95 or 0.99?

# measure distance
measure_distance = 3

# how big the contour-area/ellipse_area has to be, that the ellipse will get plotted!
min_area_percent = 0.5

# realistic_size (in pixels)
a_max = 510   # 508 calculated, =>but we need just the half of the axes it!
a_min = 105   # 108 calculated,
b_max = 320   # 318 calculated
b_min = 55   # 58 calculated
ang_max = 45    # 45 calculated


#########################
# only main_methods
threshold = 170     # 170-180 für Model mit 3 Filtern!?! Eher 170!?

