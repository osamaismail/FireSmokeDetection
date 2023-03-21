import cv2
import numpy as np
from keras.models import load_model
from checkingGPU import GPU_memory_growth


GPU_memory_growth()


# Load the trained model
model = load_model('Model.h5')

# Define the classes
classes = ["Fire", "Smoke"]

# Create a VideoCapture object to read the video
cap = cv2.VideoCapture('Video.mp4')
# cap = cv2.VideoCapture(0)	    # For camera use


# Define the filters
min_area = 150
max_area = 7000
min_aspect_ratio = 0.3
min_solidity = 0.7
margin = 10


while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 360))

    # Break the loop if the video is over
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the image to convert it to binary
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Loop through the contours
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small or large contours
        # if area < 150 or area > 7000:
        if area < min_area or area > max_area:
            continue

        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Filter out contours with a low aspect ratio
        aspect_ratio = w / h
        if aspect_ratio < min_aspect_ratio:
            continue

        # Filter out contours that are too close to the edge of the frame
        if x < margin or y < margin or x + w > frame.shape[1] - margin or y + h > frame.shape[0] - margin:
            continue

        # Filter out contours with a low solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area
        if solidity < min_solidity:

            # Extract the ROI from the grayscale image
            roi_gray = gray[y:y+h, x:x+w]

            # Resize the ROI to match the input size of the model
            roi_gray = cv2.resize(roi_gray, (224, 224))

            # Normalize the ROI
            roi_gray = roi_gray.astype('float') / 255.0

            # Reshape the ROI to match the input shape of the model
            roi_gray = np.reshape(roi_gray, (1, 224, 224, 1))

            # Make a prediction using the model
            pred = model.predict(roi_gray)

            # Get the class with the highest probability
            class_idx = np.argmax(pred[0])
            class_label = classes[class_idx]

            # Draw the bounding box and label on the frame
            if class_label in classes and pred[0][class_idx] > 0.5:
                if class_label == "Fire":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f'{class_label}:{pred[0][class_idx] * 100:.2f}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                elif class_label == "Smoke":
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{class_label}:{pred[0][class_idx] * 100:.2f}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    # Display the frame
    cv2.imshow('frame', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()

