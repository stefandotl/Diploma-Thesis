# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:39:16 2020

@author: sliebrecht
"""


import detection_methods_v2 as dm
import cv2

def detect_ellipses(img_out):
    """detects and plots the found ellipses in the given image and returns it with the found ellipses in a list"""

    unchanged_image = img_out.copy()  # needed for copying image
    color_image = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)     # needed for displaying ellipses in color

    all_contours = dm.get_contours(img_out)
    all_contours = dm.filter_area(all_contours)  # that minimal contours get filtered, so we have at least 5 point!

    center_points = []  # all center_points from the found Ellipses will be saved in this list

    # for sorting the detected ellipses by size, ellipses: [1][1] = long_axe, [1][0] = short_axe
    # because d**3 is proportional to a*b*c, where c=a, so d**3 prop to a**2 * b
    # sort_feat = lambda ellipse_feat: ellipse_feat[1][1] ** 2 * ellipse_feat[1][0]

    # all properites ((cx, cy), (short_axe, long_axe), angle) from the found Ellipses will be saved in this list
    found_ellipses = []

    for contour in all_contours:

        copy_image = unchanged_image.copy()  # copy the original image

        ############################ delete between
        # cv2.drawContours(color_image, [contour], -1, (0,0,255), -1)
        # dm.show_img(color_image, 'This Contour')
        ############################ delete between

        hull_needed = dm.hull_needed(contour, copy_image)
        # print('hull_needed', hull_needed)

        # needed for passing the same name variable, if while-loop no new_hull is found?
        new_contour = cv2.convexHull(contour)

        filled = 0

        while hull_needed:

            filled = filled + 1     # counter

            hull_image = dm.fill_hull_in_image(contour, copy_image)

            ############################ delete between
            # cv2.drawContours(color_image, [contour], -1, (0,0,255), -1)
            # dm.show_img(color_image, 'This Contour')
            ############################ delete between

            # searches new contour in the image with the filled hull!
            hull = cv2.convexHull(contour)
            new_contour = dm.find_contour(hull, hull_image)

            # checks again if it still has to search for a new hull
            hull_needed = dm.hull_needed(new_contour, copy_image)
            # print('hull_needed 2', hull_needed)

            if filled == 3:     # this process is repeated 3 times, usually it's enough to reconstruct the whole ellipse
                hull_needed = False

        else:  # löschen wenn mit while-Schleife!

            # dm.show_img(copy_image, 'decision process')
            if new_contour is not None:     # checks if contour is empty
                new_hull = cv2.convexHull(new_contour)

                if dm.realistic_ellipse(new_hull):  # vorher mit hull_contour!
                    # print('accepted')
                    ellipse = cv2.fitEllipse(new_hull)
                    if not dm.existing_ellipse(ellipse, center_points):
                        cv2.ellipse(color_image, ellipse, (0, 0, 255), 2)  # roten Ellipse brauchen kein Hull
                        dm.save_center_point(ellipse, center_points)
                        found_ellipses.append(ellipse)
                        # print(center_points)
                        # print('################')
                # else:
                # print('declined')

    # dm.write_ellipses_in_file(found_ellipses)
    # found_ellipses.sort(key=sort_feat, reverse=True)

    return color_image, found_ellipses


if __name__ == '__main__':  # Just for Testing
    # image = cv2.imread('Testbilder/9.png', 0)
    image = cv2.imread('masks/2.png', 0)
    result_image, ellipses = detect_ellipses(image)
    dm.show_img(result_image, 'End Result')
