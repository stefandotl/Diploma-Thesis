# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:41:15 2020

@author: sliebrecht
"""

import detection_methods_v3 as dm
import cv2
import numpy as np


def detect_ellipses(image):  # bilder 1, 7,8 (9)
    """detects ellipses in a given image based on the arc-length, which is not covered by other ellipses.
    This approach is based on the first method, but adjusted to real images. Works with normal contours instead of
    hull_contours.
    """

    img_out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    contours = dm.get_contours_list(image)   # contours is a list not array!
    filtered_list = dm.filter_area(contours)
    # filtered_list = dm.filter_points(filtered_list)
    filtered_list = dm.filter_shape(filtered_list)

    found_ellipses = []
    center_points = []  # all center_points from the found Ellipses will be saved in this list

    for cnt in filtered_list:
        if dm.realistic_size(cnt):  # when ellipse has realistic dimensions it will be plotted
            cnt = np.asarray(cnt)
            hull = cv2.convexHull(cnt)
            ellipse = cv2.fitEllipse(hull)  # Konturen anpassen!!
            if not dm.existing_ellipse(ellipse, center_points):
                ellipse = dm.plot_ellipse(cnt, img_out)
                dm.save_center_point(ellipse, center_points)
                found_ellipses.append(ellipse)
                # dm.show_img(img_out)

    return img_out, found_ellipses


if __name__ == '__main__':  # Just for Testing
    # image = cv2.imread('Testbilder/5.png', 0)
    image = cv2.imread('Testbilder/0.png', 0)
    # image = cv2.imread('masks/6.png', 0)
    result_image, ellipses = detect_ellipses(image)
    # print(ellipses)
    dm.show_img(result_image)


