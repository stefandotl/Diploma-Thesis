# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:32:36 2020

@author: sliebrecht
"""

import csv
import numpy as np
import matplotlib.pyplot as plt


mu_d = 1.37  # mu-diameter, calculated from the formula above
sig_d = 0.175  # sig-diameter, calculated from the formula above
lognorm_list = []
error_list = []     # calculates the error
error_list_messreihe = []

def generate_rand_list(counter):    
    for i in range(100000):
        d_orig = np.random.lognormal(mu_d, sig_d)
        lognorm_list.append(d_orig)
        
    return lognorm_list


def evaluate_detected_ellipses(name_file, name_output_image):
    # has to be opened in read mode => 'r'

    counter_lognorm = 0

    with open('detected_ellipses.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the first line

        d_list = []     # diameter
        cx_list = []    # x_coordinate of center-ellipse
        cy_list = []    # y_coordinate of center-ellipse
        a_list = []     # a_axe
        b_list = []     # b_axe
        c_list = a_list  # c_axe = a_axe
        ang_list = []   # angle

        for row in reader:
            counter_lognorm = counter_lognorm + 1
            img_name = row[0]
            diameter = float(row[1]) + 0.2
            d_list.append(diameter)
            c_x = int(row[2])
            cx_list.append(c_x)
            c_y = int(row[3])
            cy_list.append(c_y)
            a_axe = float(row[4])
            a_list.append(a_axe)
            b_axe = float(row[5])
            b_list.append(b_axe)
            angle = float(row[6])
            ang_list.append(angle)

    with open('detected_sauter.csv', 'r') as csv_sauter:
        reader_sauter = csv.reader(csv_sauter)
        next(reader_sauter, None)  # skip the first line
        sauter_list = []

        for line in reader_sauter:
            # image_name_sauter = line[0]
            sauter_diameter = round(float(line[1]), 2)
            sauter_list.append(sauter_diameter)
    try:
        with open(name_file, 'r') as csv_messreihe:
            reader_messreihe = csv.reader(csv_messreihe)
            messreihe_list = []
    
            for line in reader_messreihe:
                try:
                    diameter_string = float(line[0] + '.' + line[1])/50
                    messreihe_diameter = round(float(diameter_string), 2)
                    messreihe_list.append(messreihe_diameter)
                except:
                    pass

            
            # filtering high and low values to get more accurate results
            messreihe_list = list(filter(lambda x: 2 < x < 7, messreihe_list))
            
    except:
        pass

    lognorm_list = generate_rand_list(counter_lognorm)
    y_scale = np.arange(1.5, 7.5, 0.2)
    # plt.hist(a_list, scale_a, histtype='bar', rwidth=0.5)
    # plt.hist(b_list, scale_b, histtype='bar', rwidth=0.5)
    # plt.hist(ang_list, scale_angle, histtype='bar', rwidth=5)
    # plt.hist(sauter_list, scale_d, histtype='bar', rwidth=0.3, density=1, label="Sauter")
    
    # filtering high and low values to get more accurate results
    d_list = list(filter(lambda x: 2 < x < 7, d_list))
    lognorm_list = list(filter(lambda x: 2 < x < 7, lognorm_list))
    
    try:
        plotting_list = [d_list, lognorm_list, messreihe_list]
    except:
        plotting_list = [d_list, lognorm_list]
        
    colors = ["Ermittelte Durchmesser", "Lognormal", "ImageJ"]
    plt.xlabel('Durchmesser in mm')
    plt.xticks(np.arange(1.5, 7.5, step=0.5))
    plt.ylabel('Relative Häufigkeit')
    plt.yticks(np.arange(0, 0.65, step=0.05))
#    plt.xticks(np.arange(1, 7.0, step=0.5))
    value_bars, bins, patches = plt.hist(plotting_list, y_scale, histtype='bar', rwidth=0.5, density=1, label=colors)
        
    # zur Berechnung der Abweichung
    try:    # wenn messreihe zum vergleichen
        for i in range(len(value_bars[2])):
            z = value_bars[1][i] - value_bars[2][i]
            difference = abs(z)
            error_list_messreihe.append(difference)

        average_number_messreihe = sum(error_list_messreihe)/len(value_bars[0])
        print('Messreihe maximale Abweichung:', max(error_list_messreihe))
        print('Messreihe durchschnittliche Abweichung:', average_number_messreihe)
        
    except:
        pass
        
    finally:
        for i in range(len(value_bars[0])):
            z = value_bars[0][i] - value_bars[1][i]
            difference = abs(z)
            error_list.append(difference)
        
        average_number = sum(error_list)/len(value_bars[0])
        print('maximale Abweichung:', max(error_list))
        print('durchschnittliche Abweichung:', average_number)
    
    plt.legend(loc="best")
    plt.savefig(name_output_image, dpi=800)
    plt.show()


if __name__ == '__main__':  # Just for Testing
    evaluate_detected_ellipses('temp_imgs_new.csv', 'ellipsenverteilung_vergleich.png')
