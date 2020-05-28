# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:43:20 2020

@author: sliebrecht
"""

import os
import numpy as np
import cv2
import math

import set_values_real_images as sv_rv


def existing_ellipse(ellipse, center_points):
    """checks if the ellipse has already been plotted/saved in the center_points-list by comoaring it's center point
    to all existing center_points"""
    x_new = int(round(ellipse[0][0]))
    y_new = int(round(ellipse[0][1]))

    for point in center_points:

        x_existing = point[0]
        y_existing = point[1]
        x_dif = abs(x_new - x_existing)
        y_dif = abs(y_new - y_existing)

        distance = math.sqrt(x_dif ** 2 + y_dif ** 2)
        if distance < 5:  # if
            # print('existing Ellipse')
            return True
        else:
            return False


def get_contours_list(image):
    """searches the contours in the given image and returnes it"""

    total_list = []

    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:

        contour_list = []   # emptys the list
        hull = cv2.convexHull(contour)

        for point in hull:

            point_list = []     # emptys the list

            x = point[0][0]
            point_list.append(x)
            y = point[0][1]
            point_list.append(y)
            contour_list.append(point_list)

        if len(contour_list) > 10:  # filtert kleine konturen
            total_list.append(contour_list)

    return total_list


def plot_contours(cnt_list, file_name):
    """Plots the ellipses from given points!
    cnt_arr: is the array of all columns passed by"""
    # TODO: mit np.min, np.max, np.ptp ? etc, die maximas und minimas rausgeben und aus diesen die Ellipse plotten!
    # TODO: über mehrere Conturen verbinden!

    cnt_arr = np.asarray(cnt_list)

    img_out = cv2.imread(file_name, cv2.IMREAD_COLOR)
    cv2.drawContours(img_out, [cnt_arr], -1, (0, 0, 255), 2)
    # print(cv2.isContourConvex(cnt_arr))

    filter_shape(cnt_arr, file_name)

    show_img(img_out)


def plot_contours_hull(cnt_list, file_name):
    """Plots the ellipses from given points!
    cnt_arr: is the array of all columns passed by"""
    # TODO: mit np.min, np.max, np.ptp ? etc, die maximas und minimas rausgeben und aus diesen die Ellipse plotten!
    # TODO: über mehrere Conturen verbinden!

    img_out = cv2.imread(file_name, cv2.IMREAD_COLOR)

    cnt_arr = np.asarray(cnt_list)
    hull = cv2.convexHull(cnt_arr)

    cv2.drawContours(img_out, hull, -1, (0, 0, 255), 3)

    show_img(img_out)


# Lieber das die Konturen einzeln übergeben werden!? JA!
def plot_ellipse(contour, img_out):
    """Plots the ellipses from given points!
    cnt_arr: is the array of all columns passed by"""
    # TODO: mit np.min, np.max, np.ptp ? etc, die maximas und minimas rausgeben und aus diesen die Ellipse plotten!
    # TODO: über mehrere Conturen verbinden!

    cnt = np.asarray(contour)
    hull = cv2.convexHull(cnt)

    # cv2.drawContours(img_out, [contour], -1, (0, 0, 255), 2)
    ellipse = cv2.fitEllipse(hull)  # Konturen anpassen!!
    img_out = cv2.drawContours(img_out, hull, -1, (0, 0, 255), 3) # shows the hull points
    img_out = cv2.ellipse(img_out, ellipse, (0, 255, 0), 2)

    # filter_shape(contour, file_name)

    # show_img(img_out)

    return ellipse


def realistic_size(contour):
    """filters ellipses if they are not realistic size
        a_axe : long_axe,
        b_axe : short_axe
    """

    cnt = np.asarray(contour)

    ellipse = cv2.fitEllipse(cnt)
    # cv2.ellipse(mask, ellipse, (155, 0, 155), 2)

    a_max = sv_rv.a_max   # 508 calculated,
    a_min = sv_rv.a_min   # 108 calculated,
    b_max = sv_rv.b_max   # 318 calculated
    b_min = sv_rv.b_min  # 58 calculated
    ang_max = sv_rv.ang_max    # 45 calculated
    short_axe = ellipse[1][0]
    long_axe = ellipse[1][1]
    angle = ellipse[2] - 90     # the angle has to be -90 deg to be the same format!
    ratio = short_axe / long_axe  # for real images diffrent?

    if long_axe < a_min or long_axe > a_max:
        # print('long axe', long_axe)
        return False

    if short_axe < b_min or short_axe > b_max:
        # print('short axe', short_axe)
        return False

    if ratio > sv_rv.ratio_max or ratio < sv_rv.ratio_min:   # 0.37 # real: 0.4 bis 0.99
        # print('ratio', ratio)
        return False

    if angle < -ang_max or angle > ang_max:
        # print(angle)
        return False

    contour_area = cv2.contourArea(cnt)
    ellipse_area = round(3.141592 * (long_axe/2) * (short_axe/2), 2)

    if contour_area > ellipse_area or contour_area < sv_rv.min_area_percent * ellipse_area:
        return False

    # show_img(show_img)
    return True

def save_center_point(ellipse, center_points):
    """saves the centerpoints in a list"""
    x_coordinate = int(round(ellipse[0][0]))
    y_coordinate = int(round(ellipse[0][1]))

    center_points.append([x_coordinate, y_coordinate])


def show_img(img_out):
    """shows all windows"""
    cv2.namedWindow('Found Ellipses', cv2.WINDOW_NORMAL)
    cv2.imshow('Found Ellipses', img_out)
    cv2.waitKey()
    cv2.destroyAllWindows()


def filter_area(cnt_list):
    """filters all all_contours based on their area and returns a new list"""
    filtered_area = []  # oder lieber als set?

    for cnt_area in cnt_list:
        points = np.array(cnt_area)
        area = cv2.contourArea(points)
        # print(area)
        if area > sv_rv.min_area:      # 6000 ?
            filtered_area.append(cnt_area)

    return filtered_area


def filter_len(cnt_list):
    """filters all small all_contours and returns a new list"""

    filtered_list = []  # oder lieber als set?

    for contour in cnt_list:
        if len(contour) > 180:
            filtered_list.append(contour)

    return filtered_list


def filter_points(contour_list):
    """filters all extreme points in the hull shape by comparing the x_n coordinate with x_(n+1)
    analog: y_n with y_(n+1). if the distance [x_(n+1) or y_(n+1)] is to big,
    then the point is likely an extreme point and will be filtered!"""

    filtered_list = []

    for contour in contour_list:

        cnt_list = []

        # contour counter, i = 3 for skipping the first 2 points => doesnt help!
        i = 0

        # (len(contour)-1), because you have the compare previous with the following point,
        # but the last point has no following point!
        while i < (len(contour)-1):

            x_dif = contour[i][0] - contour[i+1][0]     # distance between x-coordinates of two points from a contour!
            y_dif = contour[i][1] - contour[i+1][1]     # distance between y-coordinates

            # distance between two consecutive points
            dist = math.sqrt(pow(x_dif, 2) + pow(y_dif, 2))
            # print(dist)

            # when step to the next x or y coordinate is to large the upcoming point is filtered!
            if dist < sv_rv.dist:
                cnt_list.append(contour[i])
                i = i + 1

            else:
                i = i + 1

                # checks if the filtered point has some neighbour points and filters them also,
                # but not more than 2 points
                dist_fil = 0
                j = i
                max_points = 1

                while dist_fil < sv_rv.dist_fil and j < (len(contour) - 1) and max_points < 2:
                    x_dif_fil = contour[j][0] - contour[j + 1][0]
                    y_dif_fil = contour[j][1] - contour[j + 1][1]
                    dist_fil = math.sqrt(pow(x_dif_fil, 2) + pow(y_dif_fil, 2))
                    max_points = max_points + 1
                    point_skipped = True
                    j = j + 1

                # if all neighbour-points have been filtered, the algortithm continues to search!
                i = j

        filtered_list.append(cnt_list)

    return filtered_list


def filter_shape(contour_list):
    """Checks if a point is outside/to far away from a fitted ellipse by comparing it's shape to the fitted ellipse"""
    filtered_list = []

    for contour in contour_list:

        if len(contour) > 5:  # you need at least 5 Points to plot an ellipse!

            cnt = np.asarray(contour)

            # creates a empty mask with x_width and y_width dimension, here the dimensions changed for testing purpose
            # then a ellipse is fitted for the correspondig contour and is drawn into the mask,
            # after that, the distance for every point from contour to the fitted ellipse is measured
            mask = np.zeros((sv_rv.y_width, sv_rv.x_width), np.uint8)
            ellipse = cv2.fitEllipse(cnt)
            mask = cv2.ellipse(mask, ellipse, 255, -1)
            ellipse_contour, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            distance_ok = True

            for point in contour:
                pt = tuple(point)
                # Except-Case for wrong plotted ellipses with center outside the image
                if len(ellipse_contour) != 0:  #
                    distance = cv2.pointPolygonTest(ellipse_contour[0], pt, True)

                    # checks if a point is to far away from the fitted ellipse
                    max_dist = sv_rv.max_dist  # filter, number, also in set_values?
                    if distance > max_dist or distance < -max_dist:
                        distance_ok = False
                        break
                else:
                    distance_ok = False

            if distance_ok:

                if cv2.contourArea(ellipse_contour[0]) > sv_rv.min_area:
                    conv_cnt = cnt.tolist()
                    filtered_list.append(conv_cnt)

    return filtered_list





