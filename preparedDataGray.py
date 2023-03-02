import json
import os
import numpy as np
import cv2


def load_annotations(json_file):
    with open(json_file, 'r') as f:
        annot = json.load(f)
    return annot

def preprocess_data(data_dir, size=(224, 224), grayscale=False):
    X = []
    y = []
    classes = {'Fire': 0, 'Smoke': 1}
    f = s = 0

    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            annot = load_annotations(os.path.join(data_dir, filename))
            ImageName = os.path.join(data_dir, annot['imagePath'])
            if os.path.exists(ImageName):
                img = cv2.imread(ImageName)
                if grayscale:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                h, w = annot['imageHeight'], annot['imageWidth']
                image_labels = []
                for shape in annot['shapes']:
                    label = shape['label']
                    if label in classes:
                        class_id = classes[label]
                        points = np.array(shape['points'], dtype=np.int32)
                        xmin, ymin = np.min(points, axis=0)
                        xmax, ymax = np.max(points, axis=0)
                        roi = img[ymin:ymax, xmin:xmax]
                        roi = cv2.resize(roi, size)
                        X.append(roi)
                        image_labels.append(class_id)
                        if label == 'Fire':
                            f += 1
                        else:
                            s += 1
                y.extend(image_labels)
            else:
                print("File not found:", ImageName)
    print('Fire:', f)
    print('Smoke:', s)
    X = np.array(X)
    y = np.array(y)
    return X, y
