import tensorflow as tf
import matplotlib.pyplot as plt
# from preparedDataGray import preprocess_data
# from getTestedImagesGray import testDataLabels
from newPreparedData import preprocess_data
from newTestedimages import testDataLabels
from checkingGPU import GPU_memory_growth
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve

GPU_memory_growth()

data_dir = "labeledImages"

# Load and preprocess the data
X, y = preprocess_data(data_dir, grayscale=True)  # set grayscale argument to True

# Split the data into train, validation, and test sets
train_split = 0.6
val_split = 0.2
test_split = 0.2

num_samples = X.shape[0]
train_samples = int(train_split * num_samples)
val_samples = int(val_split * num_samples)
test_samples = int(test_split * num_samples)

X_train, y_train = X[:train_samples], y[:train_samples]
X_val, y_val = X[train_samples:train_samples + val_samples], y[train_samples:train_samples + val_samples]
X_test, y_test = X[-test_samples:], y[-test_samples:]

# Define the input shape for the modified model
input_shape = (224, 224, 1)

# Create a new input layer with the desired shape
input_layer = tf.keras.layers.Input(shape=input_shape)

# Stack three copies of the input tensor along the channel dimension to create a 3-channel image
input_layer_stacked = tf.keras.layers.concatenate([input_layer, input_layer, input_layer], axis=3)

# Load the MobileNetV2 model with the modified input layer
base_model = tf.keras.applications.MobileNetV2(input_tensor=input_layer_stacked, include_top=False, weights='imagenet')

# Freeze the base model layers
base_model.trainable = False

# Define the custom model
model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.UpSampling2D((4,4), interpolation='nearest'),
        tf.keras.layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(512, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_data=(X_val, y_val))


# Split the test data and labels into two arrays based on the class labels
test_data_class_0 = X_test[y_test == 0]
test_labels_class_0 = y_test[y_test == 0]

test_data_class_1 = X_test[y_test == 1]
test_labels_class_1 = y_test[y_test == 1]


# Evaluate the model on each array and print the test loss and accuracy for each class separately
test_loss_class_0, test_acc_class_0 = model.evaluate(test_data_class_0, test_labels_class_0)
print("Class Fire Test Loss:", test_loss_class_0)
print("Class Fire Test Accuracy:", test_acc_class_0)

test_loss_class_1, test_acc_class_1 = model.evaluate(test_data_class_1, test_labels_class_1)
print("Class Smoke Test Loss:", test_loss_class_1)
print("Class Smoke Test Accuracy:", test_acc_class_1)

# Save the model for future use
model.save("detectGray1.h5")

# Plot the training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# Load the test data and labels
testData = 'TestImages'
test_data, test_labels = testDataLabels(testData, grayscale=True) # set grayscale argument to True

# Make predictions on the test data
predictions = model.predict(test_data)

# Calculate precision and recall values for the positive class
precision, recall, _ = precision_recall_curve(test_labels, predictions[:, 1])

# Compute the average precision (AP)
ap = average_precision_score(test_labels, predictions[:, 1])

# Print the mAP value
print(f"mAP: {ap:.2f}")

# Convert the predictions to class labels
predictions_class = np.argmax(predictions, axis=1)

# Calculate the confusion matrix
conf_mat = confusion_matrix(test_labels, predictions_class)

# Create a classification report
class_rep = classification_report(test_labels, predictions_class)

# Print the confusion matrix and classification report
print("Confusion Matrix:")
print(conf_mat)
print("\nClassification Report:")
print(class_rep)

# Save the confusion matrix and classification report to a text file
with open("CC-report.txt", "w") as f:
        f.write("Confusion Matrix:\n")
        np.savetxt(f, conf_mat, fmt="%d")
        f.write("\nClassification Report:\n")
        f.write(class_rep)
        f.write(F"\nmAP:{ap:.2f}")
