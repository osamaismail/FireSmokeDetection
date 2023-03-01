import tensorflow as tf
import matplotlib.pyplot as plt
from preparedDataGray import preprocess_data
from getTestedImagesGray import testDataLabels
from checkingGPU import GPU_memory_growth
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

GPU_memory_growth()

data_dir = "labeledImages"
# data_dir = "TestImages"

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


# Load the MobileNetV2 model
# base_model = tf.keras.applications.MobileNetV2(input_shape=(224,224,1), include_top=False, weights='imagenet')  # set input_shape to (224, 224, 1) for grayscale input

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
history = model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Loss:', test_loss)
print('Accuracy:', test_acc)

# Save the model for future use
model.save("detectGray3.h5")

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
