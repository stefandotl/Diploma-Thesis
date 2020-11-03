# Diploma-Thesis: Image segmentation for determining mass transfer surfaces in bubble columns using a neural network (public-version)
Image segmentation for determining mass transfer surfaces in bubble columns using a neural network


####################################

!Attention! This is not the full-version and therefore not working properly!
This is just an overview of the modules, some of them are not available.

####################################

How to use:

0. Go to "Modules\" directory.

1. If you don't have the "imgs" folder or images in "imgs/" directory, create images with create_image_new.py first, you can change the parameters num_imgs (number_of_images),  max_cover (ellipse-cover) and alphaT (holdup) in set_values.py

2. Execute main.py, this will segment the images.

   if you are segmentating artificial images, use first method (ed1) and comment the other methods out:
   => result_image, ellipses = ed1.detect_ellipses(transformed_image)     # first method, works best for unclear borders

   if you are segmentating real images, use third method (ed3) and comment the other methods out:
   => result_image, ellipses = ed3.detect_ellipses(transformed_image)  # third method, for real images adjusted
   (- ed2 is only for detection of ideal-segmented ellipses and usually not needed)

   -you can also modify the detecting-values in set_values.py for first method or set_values_real_images.py for third method

   The results will be splitted in:
   - segmented images in directory net_output/
   - images of the detected ellipses in directory results/
   - measured features in the file "detected_ellipses.csv"

3. For getting the size-distribution, execute "evaluate_results.py"

4. For calculating the sauter-diameter and mass transfer surface execute "evaluate_results_sauter.py"

That's it! The rest of the modules are just outsourced modules for better structure or not needed.
- You can also generate the featuremaps by executing "vis_tensor_new.py".

####################################

Modules overview:

- convert_images.py:
module converts images to distance-transformed images that can be segmentated by trained u-net model

- create_image_new.py:
creates artificial images with ellipses in determined size-distribution

- create-train-data.py:
creates dataset from artificial images. Use create_image_new.py first.

- generate_ellipses.py:
checks if an ellipse will be covered by other ellipses if this ellipse will be placed on this spot. This model is imported in the module create_image_new.py

- detection_methods.py:
methods for ellipse detection. This module is only working with ellipse_detection_v1.py

- detection_methods_v2.py:
methods for ellipse detection. This module is only working with ellipse_detection_v2.py

- detection_methods_v3.py:
methods for ellipse detection. This module is only working with ellipse_detection_v3.py

- ellipse_detection_v1.py:
main function for detection ellipses, method 1 works best for unclear border,
only working with detection_methods.py

- ellipse_detection_v2.py:
main function for detection ellipses, method 2 works best for ideal segmented ellipses,
only working with detection_methods_v2.py

- ellipse_detection_v3.py:
main function for detection ellipses, method 3 works best for real gas bubbles,
only working with detection_methods_v3.py

- evaluate_results.py:
reads all attributes from the detected_ellipses.csv and saves the visualization as an image, 
can also compare with other csv-files, for example "1-0.csv" as name_file

- evaluate_results_image_j.py:
reads all attributes from the detected_ellipses.csv and compares the results with image_j csv-file 

- evaluate_results_sauter.py:
calculates sauter diameter and volumenspezifische Stoffaustaustauschfläche

- main.py
main function for the segmentation process, this Module iterates through all images in "imgs/" folder and segment all ellipses with the trained convolutional network-model "dispersion_unet_best_model.pth". Afterwards a detection-process gets executed and the features of the detected objects get saved in a separate file. The result-images of the segmentation and detection get saved in 'net_output/' and 'results/' for manual evaluation.

- main_methods.py
contains all methods used in main.py

- set_values.py (used for artificial ellipses)
This module links all comprehensive parameters for create_images_new.py, create_train_data.py, test_result.py, main.py,  train_network.py in one file

- set_values_real_images.py (used for real images)
lists all the parameters in detection_methods_v3.py and ellipse_detection_v3.py

- train_network.py
trains the network with images. You need first to create a dataset. Use create-train-data.py first.

- video_capture.py
creates images for evaluating from a video file/stream

- vis_tensor_new.py
visualizes feature maps

Not needed:
- test_result.py ===> better use main.py 
tests the performance of the trained-network 
