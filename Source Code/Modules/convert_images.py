# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:40:30 2020

@author: sliebrecht

This module converts images to distance-transformed images that can be segmentated by trained u-net model
"""


import cv2
import os
import numpy as np


path = 'messreihe_bilder/'

try:
	os.mkdir('messreihe_bilder/')
except:
	pass

if len(os.listdir(path) ) == 0:
    raise FileNotFoundError("No Files in 'messreihe_bilder/' directory.")


def convert_images():
	"""converts images to distance-transformed images that can be segmentated by trained u-net model"""
	for i, filename in enumerate(os.listdir(path)):

	    file_path_name = "messreihe_bilder/" + filename
	    image = cv2.imread(file_path_name, 0)

	    length=len(filename)
	    filename_png = filename[:length-4:] + '.png'	# modification for getting the name

	    size_image = image.shape
	    new_image = np.zeros(size_image, np.uint8)  # Leere Segmentierungsmaske

	    _, thresh_image = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV)  # threshold is own variable

	    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	    for cnt in contours:
	        if cv2.contourArea(cnt) > 100:
	            cv2.drawContours(new_image, [cnt], -1, (255, 255, 255), -1)

	    dist = cv2.distanceTransform(new_image, cv2.DIST_L2, 3)
	    cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
	    cv2.imwrite('imgs/' + filename_png, dist)  # Speichern des Input für NN
	    # cv2.imwrite('thresh/' + filename, new_image)  # Nur zum anschauen, wird nicht benötigt!
	    print(filename, "done")

if __name__ == "__main__":
	print('Process Started')
	convert_images()
	print('Convertion finished!')