# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:47:10 2020

@author: sliebrecht

This Module iterates through all images in "imgs/" folder and segment all ellipses/gas-bubbles with \
the convolutional network-model "dispersion_unet_best_model.pth". 
Afterwards this detection-process gets executed and the features of the detected objects get saved in a separate file.
The result-images of the segmentation and detection get saved in 'net_output/' and 'results/' for manual evaluation.
"""

import cv2
import numpy as np
import torch
import time
import os


import set_values as sv
import ellipse_detection_v1 as ed1
import ellipse_detection_v2 as ed2
import ellipse_detection_v3 as ed3
import main_methods

directories = ["imgs", "net_output", "results"]
for name in directories:
    try:
        os.mkdir(name)
    except:
        pass

path = 'imgs/'
number_of_images = len(os.listdir(path))

def main():	
    """main process for automated segmentation and detection of ellipses/gas-bubbles in images"""

    # Cpu or gpu support-----------------------------------------------------
    gpuid = 0
    device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')
    model = main_methods.load_model(device, gpuid)

    # set the model to evaluation mode, since we're only generating output and not doing any back propogation
    model.eval()

    # emptys the file detected_ellipses.csv and writes the headline with the properties in it
    main_methods.write_head_line()

    time_per_image = 0  # needed for time estimation
    summed_time = 0     # needed for time estimation
    average_time = 0    # needed for time estimation
    counter = 0         # needed for time estimation

    if len(os.listdir(path) ) == 0:
        raise FileNotFoundError("No Files in 'imgs/' directory. Please create images with 'create_image_new.py' first.")

    for image_number, image_name in enumerate(os.listdir(path)):

        start_image_time = time.time()  # for measuring how long it takes

        if image_number > 0:    # displaying estimated remaining time
            estimated_time = average_time * (number_of_images - image_number)
            rounded_time = round(estimated_time/60, 1)
            print(rounded_time, "estimated minutes remaining")

        print("processing...")
        file_name = "imgs/" + image_name

        loaded_image = cv2.imread(file_name, 0) / 255  # so funktioniert es!!!!!

        # split the image, because the whole image doesn't fit in the cnn format (it has to be quadratic)!
        cropped_image = main_methods.crop_image(loaded_image)
        cropped_image = cropped_image.to(device)  # [NBATCH, 3, H, W]

        output_image = model(cropped_image)  # [NBATCH, 2, H, W]
        # output_right = model(right_side)

        output_image = output_image.detach().squeeze().cpu().numpy()  # get output and pull it to
        output_image = np.moveaxis(output_image, 0, -1)  # reshape moving last dimension

        final_image = cv2.normalize(output_image[:, :, 1], None, 0, 255, cv2.NORM_MINMAX)

        transformed_image = main_methods.transform(final_image, otsu=False)
        transformed_image = np.uint8(transformed_image)     # needed for ellipse detection

        result_image, ellipses = ed1.detect_ellipses(transformed_image)     # first method, works best for unclear borders on artificial images
        # result_image, ellipses = ed2.detect_ellipses(transformed_image)   # second method, works good for ideal borders
        # result_image, ellipses = ed3.detect_ellipses(transformed_image)  # third method, adjusted first method for real images 

        if image_number < 50:  # writes maximum 50 images to see the results
            cv2.imwrite('net_output/' + image_name, final_image)
            cv2.imwrite('results/' + image_name, result_image)

        main_methods.write_ellipses_in_file(ellipses, image_name)

        percent_done = round(image_number/number_of_images*100, 1)

        print(image_name, "processed.", percent_done, "% done")

        # the code below is just for displaying and estimating time
        if image_number == 0:   # for estimating remaining time
            meantime = time.time()
            time_per_image = (meantime - start_image_time)
            average_time = time_per_image
        else:
            meantime = time.time()
            time_per_image = (meantime - start_image_time)
            summed_time = summed_time + time_per_image
            counter = counter + 1
            if counter == 4:    # counts the average of 5 images!
                average_time = summed_time/5
                summed_time = 0
                counter = 0



if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Finished! The segmentation process for {len(os.listdir(path))} images took %.2f seconds!" % (end_time - start_time))    # Calculate the execution time and print the result
    if len(os.listdir(path)) < 1000:
        print(f"For good size-distribution-results it's recommended to use at least 1000 images, this run processed only {len(os.listdir(path))} images.")

