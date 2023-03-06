import os
import cv2
import json
import numpy as np


def testDataLabels(testPath, size=(224, 224), grayscale=False):
    # Load the test data and labels
    test_images = []
    test_labels = []
    testLabeledImg = []

    # Get the JSON file containing annotations for all the test images
    annotations_file = os.path.join(testPath, 'annotations.json')

    with open(annotations_file) as f:
        annotations = json.load(f)

    # Load the images and labels
    for annotation in annotations:
        filename = annotation['image']
        filePath = os.path.join(testPath, filename)

        # Load the image
        img = cv2.imread(filePath)
        test_images.append(img)

        # Load the corresponding labels from the annotations
        for ann in annotation['annotations']:
            label = ann['label']
            if label == 'fire':
                test_labels.append(0)
            elif label == 'smoke':
                test_labels.append(1)
            else:
                raise Exception(f"Invalid label: {label}")

            xmin = int(ann['coordinates']['x'])
            ymin = int(ann['coordinates']['y'])
            xmax = int(ann['coordinates']['x'] + ann['coordinates']['width'])
            ymax = int(ann['coordinates']['y'] + ann['coordinates']['height'])
            roi = img[ymin:ymax, xmin:xmax]
            roi = cv2.resize(roi, size)
            if grayscale:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            testLabeledImg.append(roi)

    # Convert the images and labels to NumPy arrays
    test_data = np.array(testLabeledImg)
    test_labels = np.array(test_labels)
    return test_data, test_labels
