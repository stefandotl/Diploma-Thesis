
import cv2
import numpy as np
import torch
import csv
from unet import UNet
from matplotlib import pyplot as plt
import os.path
from os import path

import set_values as sv


def load_model(device, gpuid):
    # should match the value used to train the network, will be used to load the appropirate model
    dataname = "dispersion"

    # Cpu or gpu support-----------------------------------------------------
    # print(torch.cuda.get_device_properties(gpuid))
    # torch.cuda.set_device(gpuid)
    # device = torch.device(f'cuda:{gpuid}' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(f"{dataname}_unet_best_model.pth", map_location='cpu')

    # load the model, note that the paramters are coming from the checkpoint,
    # since the architecture of the model needs to exactly match the weights saved
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                 padding=checkpoint["padding"],
                 depth=checkpoint["depth"], wf=checkpoint["wf"], up_mode=checkpoint["up_mode"],
                 batch_norm=checkpoint["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])
    return model


def merge_image():
    """merge the square splits and returns the full image"""
    pass


def show_img(img_out, window_name="Default Name"):
    """Displays a given Image, you can also pass a second argument which can be a name or a second Image as a parameter.
    """
    # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
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


def crop_image(image):
    """splits the given image (either in resolution 1024x1280, or 775x775) in square images and returns them as
    Tensors back."""

    try:    # shape of aritfical images is 1280x1024, so we need to crop a square shape (1024x1024) from it
        cropped_image = image[:, 128:1152]
        cropped_image = torch.Tensor(cropped_image).reshape(1, 1, 1024, 1024)

    except:    # the real images from the basler_camera are smaller with a resolution 775x775 or 700x700!
        dimensions = image.shape
        x = dimensions[0]
        y = dimensions[1]
        if x == y:
            cropped_image = image
            cropped_image = torch.Tensor(cropped_image).reshape(1, 1, x, y)
        else:
            raise ValueError('Wrong Image-Shape: The Shape of the image has to be quadratic or 1280x1024')

    return cropped_image


def transform(image, otsu=False):
    """transforms the image making a scale?"""

    if otsu:
        new_image = cv2.GaussianBlur(image, (5, 5), 0).astype('uint8')
        ret, thresh_image = cv2.threshold(new_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    else:
        new_image = cv2.blur(image, (5, 5))
        ret, thresh_image = cv2.threshold(new_image, sv.threshold, 255, cv2.THRESH_BINARY)  # threshold is own variable

    return thresh_image


def write_head_line():
    """emptys the file detected_ellipses.csv and writes the headline in it"""

    with open('detected_ellipses.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(["Image_name", "diameter", "Center x-coord.", "Center y-coord.",
                         "a-axis", "b-axis", "c-axis", "angle"])

def write_head_sauter():
    """emptys the file detected_sauter.csv and writes the headline in it"""

    with open('detected_sauter.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(["image_name", "Sauter Durchmesser"])


def write_ellipses_in_file(ellipse_list, image_name):
    """writes the properties of the given ellipse in a csv file, sorted by the large_axe (a_axe)
    Additionally calculates the diameter of the equivalent ellipse values.
    4/3*pi*(d/2)**3 = 4/3*pi*a*b*c, with a = c
    """

    sort_feat = lambda ellipse_feat: ellipse_feat[1][1] ** 2 * ellipse_feat[1][0]
    ellipse_list.sort(key=sort_feat, reverse=True)

    v_ges = 0   # gesamtes volumen, wird zur Berechnung des Sauter_Durchmessers gebraucht
    o_ges = 0   # gesamte Oberfläche, ebenfalls für Sauter

    with open('detected_ellipses.csv', mode='a') as file:
        writer = csv.writer(file, lineterminator='\n')
        for ellipse in ellipse_list:
            x_pix = ellipse[0][0]
            x = int(round(x_pix))
            y_pix = ellipse[0][1]
            y = int(round(y_pix))
            short_axis = ellipse[1][0]
            b = round(short_axis / 100, 5)  # short_axis/(50*2), because only half axis 
            long_axis = ellipse[1][1]   # because only the half axis is counted by us
            a = round(long_axis / 100., 5)  # converted from pixel in millimeter
            c = a
            angle = round(ellipse[2])
            converted_angle = angle - 90    # convertion, that angle has same format like from create_image_new.py

            v_ges = v_ges + (4/3)*3.1415*a*b*c
            o_ges = o_ges + 4 * 3.1415 * (((a * b) ** 1.6 + (a * c) ** 1.6 + (b * c) ** 1.6) / 3) ** 0.625

            diameter = round((8 * (a ** 2) * b) ** (1. / 3.), 2)
            # "image_name", "diameter", "Center x-coord.", "Center y-coord.", "a-axis",
            # "b-axis", "c-axis", "angle", "sauter"

            writer.writerow([image_name, diameter, x, y, a, b, a, converted_angle])

    # try:
    #     sauter = 6 * v_ges / o_ges
    #     with open('detected_sauter.csv', mode='a') as csv_sauter:
    #         writer_s = csv.writer(csv_sauter, lineterminator='\n')
    #         writer_s.writerow([image_name, sauter])
    # except:
    #     print(f'No ellipses/sauter in {image_name} found!')


if __name__ == '__main__':  # Just for Testing
    # image_name = 'netz_bilder/0_2.png'
    # image_name = 'netz_bilder/68_0.png'
    image_name = 'netz_bilder/3.png'

    image = cv2.imread(image_name, 0)
    transfromed_image = transform(image)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.imshow('test', transfromed_image)
    cv2.waitKey()

