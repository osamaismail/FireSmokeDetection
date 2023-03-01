# FireSmokeDetection
Fire &amp; Smoke Detection Using custom MobileNetV2


## 1- preparedDataGray.py

This code defines two functions, load_annotations and preprocess_data.

load_annotations function takes a JSON file name as input, reads the contents of the file and returns the JSON object. The function opens the file in read mode, loads the contents of the file using json.load method and returns the object.

preprocess_data function takes a directory path, a tuple representing image size (default is (224,224)), and a boolean grayscale flag (default is False) as inputs. The function reads each file in the directory and loads the JSON annotation data from the file using the load_annotations function. It then reads the image file mentioned in the annotation data and applies some preprocessing steps based on the provided arguments. If the grayscale flag is set to True, the function converts the image to grayscale using cv2.cvtColor method. Then, the function extracts the bounding boxes for each shape in the annotation data, crops the corresponding region of interest (ROI) from the image, resizes the ROI to the specified size using cv2.resize method, and appends the ROI to the X list. The function also creates a list of class labels for each shape in the annotation data and extends the y list with these labels. Finally, the function returns X and y as numpy arrays.

This code can be used as a preprocessing step for training a machine learning model on images with annotations. The directory containing the images and the corresponding JSON files with annotations can be passed as input to the preprocess_data function. The output of this function can be used as input to the machine learning model for training.

## 2- checkingGPU.py

This code defines a function GPU_memory_growth which sets the memory growth of the available GPUs in TensorFlow.

The function sets the environment variable TF_CPP_MIN_LOG_LEVEL to 3 to suppress TensorFlow logs. Then, it retrieves a list of physical GPUs using tf.config.list_physical_devices('GPU'). If there are available GPUs, the function sets the memory growth of each GPU using tf.config.experimental.set_memory_growth(gpu, True). This enables TensorFlow to allocate memory on the GPU as needed instead of allocating all of the available memory upfront, which can lead to out-of-memory errors. The function then prints the number of physical and logical GPUs available using tf.config.list_logical_devices('GPU').

This function can be called before running a TensorFlow script that uses GPU(s) to ensure that memory is allocated efficiently. This can be especially useful when working with large datasets or models that require a lot of GPU memory.

## 3- getTestedImagesGray.py

This code defines two functions, getDirectoriesForImages and testDataLabels.

getDirectoriesForImages function takes a directory path as input and returns a list of filenames of image files in the directory that end with .jpg, .png, or .jpeg. The function filters the list of all filenames in the directory using a list comprehension, checking for filenames that have the desired extensions. It then returns the sorted list of filenames.

testDataLabels function takes a directory path, a tuple representing image size (default is (224,224)), and a boolean grayscale flag (default is False) as inputs. The function retrieves the list of filenames of image files in the directory using the getDirectoriesForImages function. It then loads each image file, and loads the corresponding label from the JSON file associated with each image. The function creates a list of image labels based on the label found in the JSON file. If the label is 'Fire', the function appends a 0 to the list, if it is 'Smoke', it appends a 1. The function raises an exception if the label is neither 'Fire' nor 'Smoke'. The function then extracts the bounding box coordinates for the shape in the JSON file, crops the corresponding region of interest (ROI) from the image, resizes the ROI to the specified size using cv2.resize method, and applies the grayscale conversion if the grayscale flag is True. The function appends the processed ROI to the testLabeledImg list. Finally, the function returns testLabeledImg and test_labels as numpy arrays.

This code can be used to preprocess test data for a machine learning model that takes image data with annotations as input. The directory containing the test images and their corresponding JSON files can be passed as input to the testDataLabels function. The output of this function can be used as input to the machine learning model for inference.

## 4- trainModelGray.py

This code trains a machine learning model for image classification using the MobileNetV2 architecture on grayscale images with annotations.

The code starts by calling the preprocess_data function to load and preprocess the labeled images from a specified directory. The preprocessed data is then split into train, validation, and test sets. The MobileNetV2 architecture is then loaded with a modified input layer to accept grayscale images. The base model layers are frozen to prevent their weights from being updated during training. A custom model is defined on top of the base model, with additional layers for upsampling, convolution, pooling, and dense layers. The model is compiled with an Adam optimizer, sparse categorical cross-entropy loss function, and accuracy metric. The model is trained on the train and validation sets for 15 epochs with a batch size of 8. The test data is then evaluated on the trained model, and the test loss and accuracy are printed. The trained model is saved to a file.

The code then plots the training and validation accuracy and loss for each epoch using matplotlib.

The testDataLabels function is then called to load and preprocess the test images and labels. The trained model is used to make predictions on the test data, and the predictions are converted to class labels. The confusion matrix and classification report are then printed to evaluate the performance of the trained model on the test data.

Overall, this code can be used as a starting point to train a machine learning model for image classification on grayscale images with annotations. The architecture can be modified and hyperparameters can be tuned to improve performance.

## 5- FSdetectContours.py

This code detects fire and smoke in a video stream using a pre-trained machine learning model.

The code first loads the pre-trained model using load_model from keras.models and defines the classes to be detected, which are "Fire" and "Smoke". The video stream is read using cv2.VideoCapture. The video frames are then processed in a loop. Each frame is first resized and converted to grayscale. The image is thresholded to convert it to binary, and the contours are found using cv2.findContours. For each contour, the area is computed, and small or large contours are filtered out. The bounding rectangle of the contour is extracted, and the region of interest (ROI) is cropped from the grayscale image. The ROI is then resized and normalized to match the input size and range of the model. A prediction is then made using the model, and the class with the highest probability is determined. If the class label is either "Fire" or "Smoke" and the probability is greater than 0.9, the bounding box and label are drawn on the frame. The processed frame is displayed, and the loop continues until the 'q' key is pressed, at which point the video stream is released and the display window is closed.

Overall, this code demonstrates how a pre-trained machine learning model can be used for real-time fire and smoke detection in a video stream using computer vision techniques. The model can be modified and tuned to improve its accuracy and performance.
