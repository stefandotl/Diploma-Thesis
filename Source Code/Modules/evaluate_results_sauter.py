# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 12:32:36 2020

@author: sliebrecht
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

def evaluate_detected_ellipses():
    """ """

    with open('detected_ellipses.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)  # skip the first line
        
        hold_up = 0.1
        v_ges = 0   # gesamtes volumen, wird zur Berechnung des Sauter_Durchmessers gebraucht
        o_ges = 0   # gesamte Oberfläche, ebenfalls für Sauter

        for row in reader:
            diameter = float(row[1]) + 0.2
            
            v_ges = v_ges + (4/3)*3.1415*(diameter/2)**3
            o_ges = o_ges + 4 * 3.1415 *(diameter/2)**2
            
        sauter = 6*v_ges/o_ges 
        print('sauter:', sauter)
        sauter = sauter * 0.001
        
        spez_flaeche = (6*hold_up)/(sauter*(1-hold_up))
        print('Stoffspezifische Flaeche', spez_flaeche)

if __name__ == '__main__':  # Just for Testing
    evaluate_detected_ellipses()
