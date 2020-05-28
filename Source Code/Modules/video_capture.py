# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:57:29 2020

@author: sliebrecht
"""
import cv2
import numpy as np



try:
    os.mkdir("video_images")
except:
    pass

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
# cap = cv2.VideoCapture('videos/ug_1.avi')   # this is for recorded video
cap = cv2.VideoCapture(0)     # this is for live camera

# Check if camera opened successfully
if (cap.isOpened() == False):
    raise RuntimeError("No video stream or file found")

seconds = 2     # how many seconds it shall wait to take an image
fps = cap.get(cv2.CAP_PROP_FPS)  # Gets the frames per second
multiplier = fps * seconds

counter = 0     # counting frames
image_counter = 0   # counting images
# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    counter = counter + 1

    ret, frame = cap.read()
    cv2.imshow('Frame', frame)

    if counter % multiplier == 0:
        image_counter = image_counter + 1
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # crop the image, that the black circle isn't visible
        try:
            cropped_image = gray_img[80:855, 253:1028]     # recommended crop: [80:855, 253:1028]
            cv2.imwrite(f"video_images/{image_counter}.png", cropped_image)
        except:
            cv2.imwrite(f"video_images/{image_counter}.png", gray_img)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

print("Video Done!")
