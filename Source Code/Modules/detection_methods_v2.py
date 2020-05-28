# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:42:44 2020

@author: sliebrecht
"""

import os
import numpy as np
import cv2
import csv
import math

import set_values as sv


def get_contours(image):
    """searches the contours in the given image and returnes it"""

    # vielleicht das versuchen! cv2.CHAIN_APPROX_TC89_L1
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours


def save_center_point(ellipse, center_points):
    """saves the centerpoints in a list"""
    x_coordinate = int(round(ellipse[0][0]))
    y_coordinate = int(round(ellipse[0][1]))

    center_points.append([x_coordinate, y_coordinate])


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


def fill_hull_in_image(contour, image):
    """fill the hull contour in the given Image"""

    # show_img(image)
    if contour is not None:
        hull = cv2.convexHull(contour)
        image = cv2.drawContours(image, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        # img_out = cv2.drawContours(image, [hull], -1, (0, 255, 255), thickness=cv2.FILLED)  # for better visualization

    return image


def filter_area(contours):
    """filters all small(unrealtistic) contours based on their area and returns a new list"""
    filtered_area = []  # oder lieber als set?

    for cnt_area in contours:
        area = cv2.contourArea(cnt_area)
        # print(area)
        if area > sv.min_area:  # 6000 ?
            filtered_area.append(cnt_area)

    return filtered_area


def find_contour(hull_points, hull_image):
    """searches the new given contour which results by filling the hull,
     based on the same shared points of both contours"""

    new_contours, hierarchy = cv2.findContours(hull_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    hull_points = hull_points.tolist()

    for contour in new_contours:

        contour_list = contour.tolist()  # you have to convert this to a list, otherwise, it won't work correctly!

        for point in hull_points:

            # if a point of the previous contour has been found in the new contour, it means,
            # it has found the expanded contour
            if point in contour_list:
                return contour


def hull_needed(contour, copy_image):
    """checks if making a hull would result in a bigger area, if yes, the chances are high to get a complete ellipse.
    If no, making a hull won't help to find the correspongig ellipse and the contour gets discarded.
    The comparsion is based on the area before and after the contour is processed with fill_hull() """

    hull_image = fill_hull_in_image(contour, copy_image)

    if contour is not None:
        hull = cv2.convexHull(contour)  # needed that we have less points to tierate through
        found_contour = find_contour(hull, hull_image)

        new_contour_area = cv2.contourArea(found_contour)
        previous_contour_area = cv2.contourArea(contour)

        #################### delete
        # cv2.drawContours(hull_image, [found_contour], -1, (255, 255, 0), -1)  # do i need this line?
        # show_img(hull_image, 'found Contour, in hull needed')
        #################### delete

        ratio = previous_contour_area / new_contour_area
        # print(ratio)

        if ratio < sv.ratio:
            return True
        else:
            return False
    else:
        return False


def measure_distance(contour):
    """measure the distance for every Point from the hull contour to the plotted ellipse"""

    # creates a empty mask with x_width and y_width dimension
    # then a ellipse is fitted for the correspondig contour and drawn into the mask
    hull_contour = cv2.convexHull(contour)
    mask = np.zeros((sv.y_width, sv.x_width), np.uint8)
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(mask, ellipse, (255, 255, 255), -1)  # draws the ellipse in the empty mask as the only contour
    ellipse_contour, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    plotted_ellipse_contour = ellipse_contour[0]  # [0] because only one contour exists

    # keep the contour
    keep = True

    for point in hull_contour:

        # oder andersrum?
        pt_x = point[0][0]
        pt_y = point[0][1]
        distance = cv2.pointPolygonTest(plotted_ellipse_contour, (pt_x, pt_y), True)
        if distance < -sv.measure_distance or distance > sv.measure_distance:
            keep = False
            break

    if not keep:
        return False
    else:
        return True


def realistic_ellipse(contour):
    """checks if the plotted ellipse is realistic by it's dimensions
    and it's points distance to the corresponding contour"""

    real_size = realistic_size(contour)
    if not real_size:
        # print('no real size')
        return False

    distance_ok = measure_distance(contour)  # measures the distance, but is it needed?´´

    if not distance_ok:
        # print('distance not okay')
        return False

    return True


def realistic_size(contour):
    """filters ellipses if they are not realistic size
        a_axis : long_axe,
        b_axis : short_axe
    """

    ellipse = cv2.fitEllipse(contour)

    a_max = sv.a_max   # 508 calculated,
    a_min = sv.a_min   # 108 calculated,
    b_max = sv.b_max   # 318 calculated
    b_min = sv.b_min  # 58 calculated
    ang_max = sv.ang_max    # 45 calculated
    short_axe = ellipse[1][0]
    long_axe = ellipse[1][1]
    angle = ellipse[2] - 90  # the angle has to be -90 deg to be the same format!
    ratio = short_axe / long_axe  # must be between 0.4-0.99

    if long_axe < a_min or long_axe > a_max:
        # print('long axe', long_axe)
        return False

    if short_axe < b_min or short_axe > b_max:
        # print('short axe', short_axe)
        return False

    if ratio > 1 or ratio < 0.37:
        # print('ratio', ratio)
        return False

    if angle < -ang_max or angle > ang_max:
        # print(angle)
        return False

    # show_img(show_img)
    return True


def filter_points(contour):
    """filters all extreme points in the hull shape by comparing the x_n coordinate with x_(n+1)
    analog: y_n with y_(n+1). if the distance [x_(n+1) or y_(n+1)] is to big,
    then the point is likely an extreme point and will be filtered!"""

    contour_filtered = []

    # contour counter, i = 3 for skipping the first 2 points => doesnt help!
    i = 0

    # (len(contour)-1), because you have the compare previous with the following point,
    # but the last point has no following point!
    while i < (len(contour) - 1):

        # nicht andersrum? x_dif = contour[i][0][1] ?
        x_dif = contour[i][0][1] - contour[i + 1][0][1]  # distance between x-coordinates of two points from a contour!
        y_dif = contour[i][0][0] - contour[i + 1][0][0]  # distance between y-coordinates

        # distance between two consecutive points
        dist = math.sqrt(pow(x_dif, 2) + pow(y_dif, 2))
        # print(dist)

        # when step to the next x or y coordinate is to large the upcoming point is filtered!
        if dist < sv.dist:
            contour_filtered.append(contour[i][0].tolist())
            i = i + 1

        else:
            i = i + 1

            # checks if the filtered point has some neighbour points and filters them also, but not more than 2 points
            dist_fil = 0
            j = i
            max_points = 1

            while dist_fil < sv.dist_fil and j < (len(contour) - 1) and max_points < 2:
                x_dif_fil = contour[j][0][1] - contour[j + 1][0][1]
                y_dif_fil = contour[j][0][0] - contour[j + 1][0][0]
                dist_fil = math.sqrt(pow(x_dif_fil, 2) + pow(y_dif_fil, 2))
                max_points = max_points + 1
                j = j + 1

            # if all neighbour-points have been filtered, the algortithm continues to search!
            i = j

    # print(type(contour_filtered))
    return np.asarray(contour_filtered)


def show_img(img_out, window_name="Default Name"):
    """Displays a given Image, you can also pass a second argument which can be a name or a second Image as a parameter.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img_out)
    cv2.waitKey()
    cv2.destroyAllWindows()


def show_two_imgs(img_out, img_two):
    """Displays a given Image, you can also pass a Name as a parameter.
    You can pass a second image aswell."""

    cv2.namedWindow("Image One", cv2.WINDOW_NORMAL)
    cv2.imshow("Image One", img_out)
    cv2.namedWindow("Image Two", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Two", img_out)
    cv2.waitKey()
    cv2.destroyAllWindows()


def write_ellipses_in_file(ellipse_list):
    """writes the properties of the given ellipse in a csv file.
    Additionally calculates the diameter of the equivalent ellipse values.
    4/3*pi*(d/2)**3 = 4/3*pi*a*b*c, with a = c
    """

    for ellipse in ellipse_list:
        x_pix = ellipse[0][0]
        x = int(round(x_pix))
        y_pix = ellipse[0][1]
        y = int(round(y_pix))
        short_axis = ellipse[1][0] / 2
        b = round(short_axis / 50., 2)
        long_axis = ellipse[1][1] / 2  # because only the half axis is counted by us
        a = round(long_axis / 50., 2)  # converted from pixel in millimeter
        angle = round(ellipse[2])
        converted_angle = angle - 90    # convertion, that angle has same format like from create_image_new.py

        diameter = round((8 * (a ** 2) * b) ** (1. / 3.), 2)

        with open('detected_ellipses.csv', mode='a') as file:
            writer = csv.writer(file, lineterminator='\n')
            # "diameter", "Center x-coord.", "Center y-coord.", "a-axis", "b-axis", "c-axis", "angle"
            writer.writerow([diameter, x, y, a, b, a, converted_angle])


########################
# not used from here

def filter_len(cnt_list):
    """filters all small all_contours and returns a new list"""

    filtered_list = []  # oder lieber als set?

    for contour in cnt_list:
        if len(contour) > 180:
            filtered_list.append(contour)

    return filtered_list


def filter_shape(contour_list):
    """Checks if a point is outside/to far away from a fitted ellipse by comparing it's shape to the fitted ellipse"""

    filtered_list = []

    for contour in contour_list:

        # print(contour)

        cnt = np.asarray(contour)

        # creates a empty mask with x_width and y_width dimension, here the dimensions changed for testing purpose
        # then a ellipse is fitted for the correspondig contour and is drawn into the mask,
        # after that, the distance for every point from contour to the fitted ellipse is measured
        mask = np.zeros((sv.y_width, sv.x_width), np.uint8)
        ellipse = cv2.fitEllipse(cnt)
        mask = cv2.ellipse(mask, ellipse, 255, -1)
        ellipse_contour, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        distance_ok = True

        for point in contour:
            pt = tuple(point)
            distance = cv2.pointPolygonTest(ellipse_contour[0], pt, True)
            # print(distance)

            # checks if a point is to far away from the fitted ellipse
            max_dist = sv.max_dist  # filter, number, also in set_values?
            if distance > max_dist or distance < -max_dist:
                distance_ok = False
                break

        if distance_ok:

            # convert numpy array to list!
            if cv2.contourArea(ellipse_contour[0]) > sv.min_area:
                conv_cnt = cnt.tolist()
                filtered_list.append(conv_cnt)

    return filtered_list
