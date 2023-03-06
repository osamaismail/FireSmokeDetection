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
    Y = []
    classes = {'fire': 0, 'smoke': 1}
    f = s = 0

    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            annot = load_annotations(os.path.join(data_dir, filename))
            for item in annot:
                ImageName = os.path.join(data_dir, item['image'])
                if os.path.exists(ImageName):
                    img = cv2.imread(ImageName)
                    if grayscale:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    for annotation in item['annotations']:
                        label = annotation['label']
                        if label in classes:
                            class_id = classes[label]
                            x, y, w, h = annotation['coordinates']['x'], annotation['coordinates']['y'], annotation['coordinates']['width'], annotation['coordinates']['height']
                            roi = img[int(y):int(y+h), int(x):int(x+w)]
                            roi = cv2.resize(roi, size)
                            X.append(roi)
                            Y.append(class_id)
                            if label == 'fire':
                                f += 1
                            else:
                                s += 1
                else:
                    print("File not found:", ImageName)
    print('Fire:', f)
    print('Smoke:', s)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y
