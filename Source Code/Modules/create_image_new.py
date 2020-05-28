# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:41:48 2020

@author: sliebrecht

This module creates artificial images with ellipses in a determined size-distribution.
"""

import random
import cv2
import numpy as np
import time
import math
import sys
import csv
import os

# interne Imports
import set_values as sv
import generate_ellipses as ge

"""Generiert weiße Ellipsen und zeichnet schwarzen Rand:
Volumen Kugel und Volumen Ellipse gleichsetzen!
V_c = V_e => 4/3*pi*r_mm**3 = 4/3*pi*a_axis*b_axis*c_axis, mit a_axis=c_axis und kürzen => r_mm**3 =  a_axis**2*b_axis (1)
ausserdem gilt: b_axis/a_axis = randrange(0.4, 0.99) => b_axis=a_axis*randrange(0.4, 0.99)
In (1) einsetzen => r_mm**3 = a_axis**3 * randrange(0.4, 0.99)
"""

directories = ["imgs", "masks", "temp_imgs"]

for name in directories:
    try:
        os.mkdir(name)
    except:
        pass

# Store start time
start_time = time.time()

number_of_images = sv.num_imgs
x_width = sv.x_width  # Bildbreite (Basler Kamera)
y_width = sv.y_width  # Bildhöhe
max_cover = sv.max_cover  # how much percent can an ellipse cover other ellipses (example: 0.3 = 30 %)
sort_feat = lambda diameter: diameter[0]  # on which sort_feat you want to sort (diameter, a_axis, b_axis, c_axis, angle)

# Variabler Sondenabstand !!!!Problem kann zu rein Schwarzen Bildern führen bei entsprechendem Phasenanteil
# space=random.randrange(10,20,1)*0.1
space = 15  # in millimeters

# Volumen in m zwischen den Sonden aufgespannt von Bildeebene und Sondenabstand
vol = (space * x_width * 0.02 * y_width * 0.02) * 0.001  # 0.02mm/pixel ... in m^3 umgerechnet

# get point random seed so that we can reproducibly do the cross validation setup
seed = random.randrange(sys.maxsize)
random.seed(seed)  # set the seed
print(seed)

# CSV datei mit Daten Random Seed und Daten der Lognormalverteilung je Bild
with open('ellipses.csv', mode='w') as csv_input:
    writer = csv.writer(csv_input, lineterminator='\n')

    # CSV Datei Schreiben mit realen Durchmesserwerten, um Zeichen algorhitmus zu überprüfen
    with open('targetdata.csv', mode='w') as csv_target:
        writer_T = csv.writer(csv_target, lineterminator='\n')
        writer_T.writerow(
            ['Random seed:', seed])  # Speichern falls exakt gleiche Bilder nochmals generiert werden müssen

        writer.writerow(["image_name", "diameter", "Center x-coord.", "Center y-coord.",
                         "a-axis", "b-axis", "c-axis", "angle"])

        print("processing...")
        counter_images = 0

        with open('sauter.csv', mode='w') as csv_sauter:
            writer_s = csv.writer(csv_sauter, lineterminator='\n')

            while counter_images <= number_of_images:

                # create / clears images
                # x and y are swapped, cause first coordinate with np.zeros is y
                img = np.zeros((y_width, x_width))  # Leeres Bild
                mask = np.zeros((y_width, x_width), np.uint8)  # Leere Segmentierungsmaske

                # m = 4;
                # v = 0.5;
                # mu = log((m ^ 2) / sqrt(v + m ^ 2));
                # sigma = sqrt(log(v / (m ^ 2) + 1));
                mu_d = 1.37  # mu-diameter, calculated from the formula above
                sig_d = 0.175  # sig-diameter, calculated from the formula above
                mu_a = 0  # mu-angle
                sig_a = 15  # sig-angle

                o_ges = 0  # Temp Surface
                v_tmp = 0  # Temp Volumen
                v_ges = 0  # Ellipsen, die nicht gefiltert wurden
                d_vol = 0  # Gesamt Volumen
                d_ob = 0  # Gesamt Oberfläche
                alpha = 0  # Am Anfang des Bildes phasenanteil auf 0 setzen

                alphaT = sv.alphaT  # maximaler Phasenanteil!

                x_tmp = []
                y_tmp = []
                a_axis_tmp = []
                b_axis_tmp = []
                c_axis_tmp = []  # In die Bildebene hinein, wird nur zur Berechnung des Volumens erstellt
                angle_tmp = []

                list_ellipse_features = []
                ellipse_counter = 0

                while alpha < alphaT:
                    # generiert Koordinaten und größe der Ellipsen
                    d = np.random.lognormal(mu_d, sig_d)
                    r_mm = 0.5 * d
                    rand_nmb = round(random.uniform(0.40, 0.99), 2)  # look docstring: rand_nmb = randrange(0.4, 0.99)
                    # in mm braucht man nicht zu runden!
                    a_axis = round(r_mm * pow(rand_nmb, -(1 / 3)), 3)  # a-axis (long-axis) in millimeters
                    c_axis = round(a_axis, 3)  # perpendicular to image plane, not visible in images!
                    b_axis = round(a_axis * rand_nmb, 3)  # short axis
                    angle = round(random.normalvariate(mu_a, sig_a))

                    ellipse_feautres = (round(r_mm, 2), round(rand_nmb, 2), a_axis, b_axis, c_axis, round(angle, 2))

                    list_ellipse_features.append(ellipse_feautres)

                    # Volumen eines Ellipsoids berechnen, in meter umrechnen!
                    v_tmp = v_tmp + 4 / 3 * math.pi * a_axis * b_axis * c_axis * 0.001
                    # v_tmp = 0.001 * v_tmp   # realistisch, oder mit 0.001**3, da alles in mm ?
                    alpha = v_tmp / vol  # Phasenanteil berechnen

                list_ellipse_features.sort(key=sort_feat, reverse=True)
       
                for i in range(len(list_ellipse_features)):
                    r_mm = list_ellipse_features[i][0]
                    rand_nmb = list_ellipse_features[i][1]
                    # die pixel Umrechnung nach der Sortierung verschieben!
                    r_pix = int(round(50 * r_mm, 2))  # transformed in pixel
                    a_axis_pix = r_pix * pow(rand_nmb, -(1 / 3))  # a-axis in pixel coordinates
                    c_axis_pix = round(a_axis_pix)  # die Achse c_axis steht senkrecht zur Bildebe!
                    b_axis_pix = round(a_axis_pix * rand_nmb)
                    a_axis_pix = round(a_axis_pix)
                    angle = list_ellipse_features[i][5]

                    # checks if ellipse is not covered, if an ellipse is covered, a new position is generalized,
                    # if no position is found without being completely covered a error window should pop up
                    for f in range(50):  # f is the number of try to find a new poistion
                        if f == 49:
                            raise ValueError("Phasenanteil zu hoch oder Überschneidung zu niedrig! "
                                             "Ellipse kann nicht platziert werden,"
                                             " ohne verdeckt zu werden!")
                        x = random.randrange(0, x_width, 1)
                        y = random.randrange(0, y_width, 1)

                        if not ge.covered_ellipse(img, a_axis_pix, b_axis_pix, x, y, angle, max_cover):
                            # cv2.namedWindow("Test", cv2.WINDOW_AUTOSIZE)
                            # cv2.imshow("Test", img)
                            # cv2.waitKey()

                            # erstellt weiße Ellipse auf maske
                            cv2.ellipse(mask, (x, y), (a_axis_pix, b_axis_pix), angle, 0, 360, 255,
                                        -1)  # only integers!?

                            # Kreis auf Inputbild malen weiß und gefüllt
                            cv2.ellipse(img, (x, y), (a_axis_pix, b_axis_pix), angle, 0, 360, 255, -1,
                                        lineType=cv2.LINE_AA)

                            x_tmp.append(x)
                            y_tmp.append(y)
                            a_axis_tmp.append(a_axis_pix)
                            b_axis_tmp.append(b_axis_pix)
                            c_axis_tmp.append(c_axis_pix)
                            angle_tmp.append(angle)

                            a_axis_mm = a_axis_pix / 50
                            b_axis_mm = b_axis_pix / 50
                            c_axis_mm = c_axis_pix / 50

                            v_ges = v_ges + (4 / 3) * 3.14159265 * a_axis_mm * b_axis_mm * c_axis_mm
                            o_ges = o_ges + 4 * 3.14159265 * (((a_axis * b_axis) ** 1.6 + (a_axis * c_axis) ** 1.6 +
                                                               (b_axis * c_axis) ** 1.6) / 3) ** 0.625

                            ellipse_counter += 1

                            image_name = str(counter_images + 1) + ".png"

                            writer.writerow([image_name, (round(2 * r_mm, 2)), x, y, str(a_axis_mm), str(b_axis_mm),
                                             str(c_axis_mm), str(angle)])  # Schreiben in die CSV

                            break

                sauter = 6 * v_ges / o_ges  # formula of sauter
                writer_s.writerow([image_name, sauter])

                # zeichnet schwarze Umrandung
                for v in range(ellipse_counter):
                    cv2.ellipse(mask, (x_tmp[v], y_tmp[v]), (a_axis_tmp[v], b_axis_tmp[v]), angle_tmp[v], 0, 360, 0, 2)

                counter_images += 1

                print("Images generated:", counter_images, "/", number_of_images,
                      "\nprocessing...", round(100 * (counter_images / number_of_images), 1),
                      "%")  # round on first number

                cv2.imwrite('temp_imgs/' + str(counter_images) + ".png", img)  # Speichern Vergleichsbild
                cv2.imwrite('masks/' + str(counter_images) + ".png", mask)  # Speichern des Groundtruth (Ziel für NN)

                # Hack um Formatumwandlung zu umgehen, einfach aber geht sicher schöner
                img_nn = cv2.imread('temp_imgs/' + str(counter_images) + ".png", 0)

                # Threshold des weißen Bilds (Nötige Umwandlung in binärbild)
                ret, th = cv2.threshold(img_nn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Distancetransform um Input für neuronales Netz zu erstellen
                dist = cv2.distanceTransform(th, cv2.DIST_L2, 3)
                cv2.normalize(dist, dist, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite('imgs/' + str(counter_images) + ".png", dist)  # Speichern des Input für NN

        print("Finished")

# Store end time
end_time = time.time()

# Calculate the execution time and print the result
print("%.10f seconds" % (end_time - start_time))
