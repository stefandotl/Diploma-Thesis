# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:15:24 2020

@author: sliebrecht

This module checks if an ellipse will be covered by other ellipses if this ellipse will be placed on this spot.
"""

import math
import numpy as np
import cv2


def covered_ellipse(img, a_axe, b_axe, cx, cy, angle, max_cover):
    """Counts the number of white pixels where the new ellipse shall be placed. If all pixels are white,
    then the ellipse would be complete covered in the shadow of other ellipses.
    That means, that the new ellipse would be undetectable and will therefore be discarded.

    img: image, where the method shall be executed, np.darray
    a_axis: a_axis of the ellipse
    b_axis: b_axis of the ellipse
    cx: x-coordinate of the center of the ellipse
    cy: y-coordinate of the center of the ellipse
    angle: the angle in Degrees with which, the ellipse is rotated
    max_cover: with how much percent can an ellipse cover other ellipses (example: 0.3 = 30 %)
    """

    teta = angle * 3.14 / 180  # Umrechnung Winkel in Bogenmaß
    count_white = 0    # counter_white_pixels
    count_black = 0    # counter_black_pixels
    # c_list = list()        # coordinate_list, only needed for Visualization

    # calcluates through every pixel around the center Point with coordinates (cx,cy)
    for x in range(cx-a_axe, cx+a_axe):
        for y in range(cy-b_axe, cy+b_axe):     # cy+b_axis darf nicht größer als y_width sein!!!
            dx = x - cx
            dy = y - cy
            x_new = (math.cos(teta) * dx - math.sin(teta) * dy) + cx      # swap x_new and y_new?
            y_new = (math.sin(teta) * dx + math.cos(teta) * dy) + cy

            distance = (dx / a_axe) ** 2 + (dy / b_axe) ** 2    # = 1 , Formula of an ellipse
            if distance <= 1:       # considers only pixels inside of the ellipse
                # check if pixel is white (white=255)
                # x_width=1280, y_width=1024 ,That no Pixels won't be drawn behind borders
                if (0 < x_new < 1279) and (0 < y_new < 1023):           # look which coordinates ar right
                    if img[round(y_new)][round(x_new)] == 255:          # first y, then x because of np.zeros format
                        count_white += 1
                        # c_list.append((round(y_new), round(x_new)))    # only needed for Visualization
                    else:
                        count_black += 1
                        # c_list.append((round(y_new), round(x_new)))    # only needed for Visualization

    # "!WARNING!: only needed for Test/Visualization, don't use to create images"!
    # for usage uncomment c_list and c_list.append..
    # for i in c_list:
    #     print("!WARNING!: only needed for Test/Visualization, don't use to create images!")
    #     img[i[0]][i[1]] = 125


    # checks how much area of the ellipse would be covered
    if count_white+count_black != 0:
        ratio = count_white/(count_white+count_black)
        # You can choose how much an ellipse can be covered to get discarded
        if ratio >= max_cover:     # example: 0.6 means the ellipse would cover 80% of other ellipses
            return True
        else:
            False
    else:
        return False


# def save_ellipse_pixels(img, a_axe, b_axe, cx, cy, angle):
#       # fixme: function is not finished
#     """Saves all Pixel from an ellipse"""
#
#     teta = angle * 3.14 / 180  # Umrechnung Winkel in Bogenmaß
#     count_white = 0  # counter_white_pixels
#     count_black = 0  # counter_black_pixels
#     c_list = set()  # coordinate_list, only needed for Visualization
#
#     for x in range(cx - a_axe, cx + a_axe):
#         for y in range(cy - b_axe, cy + b_axe):  # cy+b_axis darf nicht größer als y_width sein!!!
#             dx = x - cx     # swap x_new and y_new?
#             dy = y - cy
#             x_new = (math.cos(teta) * dx - math.sin(teta) * dy) + cx
#             y_new = (math.sin(teta) * dx + math.cos(teta) * dy) + cy
#             distance = (dx / a_axe) ** 2 + (dy / b_axe) ** 2  # = 1 , Formula of an ellipse
#             if distance <= 1:  # considers only pixels inside of the ellipse
#                 # check if pixel is white (white=255)
#                 # x_width=1280, y_width=1024 ,That no Pixels won't be drawn behind borders
#                 if (0 < x_new < 1279) and (0 < y_new < 1023):  # look which coordinates ar right
#                     if img[round(y_new)][round(x_new)] == 255:  # first y, then x because of np.zeros format
#                         count_white += 1
#                         c_list.add((round(y_new), round(x_new)))    # only needed for Visualization
#                     else:
#                         count_black += 1
#                         c_list.add((round(y_new), round(x_new)))    # only needed for Visualization
#
#     # for i in c_list:
#     #     print("!WARNING!: only needed for Test/Visualization, don't use to create images!")
#     #     img[i[0]][i[1]] = 125
#
#     return c_list


# def bresenham_ellipse(img, cx, cy, a_axis, b_axis):
#
#     def plot4_ellipse_points(x, y):
#         for i in range(x+1):    # damit alle Pixel innerhalb der Ellipse weiß sind
#             img[cy + y][cx + i] = 255    # point in quadrant 1
#             img[cy + y][cx - i] = 255    # point in quadrant 2
#             img[cy - y][cx - i] = 255    # point in quadrant 3
#             img[cy - y][cx + i] = 255    # point in quadrant 4
#
#     two_a_square = 2 * a_axis * a_axis
#     two_b_square = 2 * b_axis * b_axis
#     x = a_axis
#     y = 0
#     x_change = b_axis * b_axis * (1 - 2 * a_axis)
#     y_change = a_axis * a_axis
#     ellipse_error = 0
#     stopping_x = two_b_square * a_axis
#     stopping_y = 0
#
#     while stopping_x >= stopping_y:    # 1st set of points,  y'> -1
#         plot4_ellipse_points(x, y)
#         y += 1
#         stopping_y += two_a_square
#         ellipse_error += y_change
#         y_change += two_a_square
#
#         if (2 * ellipse_error + x_change) > 0:
#             x -= 1
#             stopping_x -= two_b_square
#             ellipse_error += x_change
#             x_change += two_b_square
#
#     # 1st point set is done; start the 2nd set of points
#     x = 0
#     y = b_axis
#     x_change = b_axis * b_axis
#     y_change = a_axis * a_axis * (1 - 2 * b_axis)
#     ellipse_error = 0
#     stopping_x = 0
#     stopping_y = two_a_square * b_axis
#     while stopping_x <= stopping_y:  # 2nd set of points, y' < -1
#         plot4_ellipse_points(x, y)
#         x += 1
#         stopping_x += two_b_square
#         ellipse_error += x_change
#         x_change += two_b_square
#
#         if (2 * ellipse_error + y_change) > 0:
#             y -= 1
#             stopping_y -= two_a_square
#             ellipse_error += y_change
#             y_change += two_a_square
#
#     cv2.namedWindow("test", cv2.WINDOW_AUTOSIZE)
#     cv2.imshow("test", img)
#     cv2.waitKey()