import os
import cv2
import json
import numpy as np


def getDirectoriesForImages(testPath):
    # Get the directory path containing the test images
    test_image_dir = testPath

    # Get the filenames of the test images
    test_filenames = os.listdir(test_image_dir)

    # Filter only the .jpg or .png files
    test_filenames = [filename for filename in test_filenames if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')]

    # Sort the filenames
    return test_filenames

def testDataLabels(testPath, size=(224, 224), grayscale=False):
    # Load the test data and labels
    test_images = []
    test_labels = []
    testLabeledImg = []

    # Get the filenames of all the test images and their corresponding labels
    test_filenames = getDirectoriesForImages(testPath)  # Get the filenames of your test images

    # Load the images and labels
    for filename in test_filenames:
        filePath = os.path.join(testPath, filename)

        # Load the image
        img = cv2.imread(filePath)
        # if grayscale:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_images.append(img)

        # Load the corresponding label from the JSON file
        json_file = filename.replace('.jpg', '.json')
        jsonPath = os.path.join(testPath, json_file)

        with open(jsonPath) as f:
            data = json.load(f)
        label = data['shapes'][0]['label']
        if label == 'Fire':
            test_labels.append(0)
        elif label == 'Smoke':
            test_labels.append(1)
        else:
            raise Exception(f"Invalid label: {label}")

        points = np.array(data['shapes'][0]['points'], dtype=np.int32)
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)
        roi = img[ymin:ymax, xmin:xmax]
        roi = cv2.resize(roi, size)
        if grayscale:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # roi = cv2.resize(roi, size)
        testLabeledImg.append(roi)

    # Convert the images and labels to NumPy arrays
    test_data = np.array(testLabeledImg)
    test_labels = np.array(test_labels)
    return test_data, test_labels
