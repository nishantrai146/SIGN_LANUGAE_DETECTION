import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load training data
with open("train_images", "rb") as f:
    train_images = pickle.load(f)
with open("train_labels", "rb") as f:
    train_labels = pickle.load(f)

# Placeholder values for image dimensions
image_y, image_x = 50, 50

# Placeholder value for the number of classes
num_classes = 10  # Update this with the actual number of gesture classes in your dataset

# Normalize pixel values to range [0, 1]
train_images = np.array(train_images) / 255.0
train_labels = to_categorical(train_labels, num_classes=num_classes)  # Convert labels to one-hot encoding

# Reshape images to (num_samples, image_height, image_width, num_channels)
train_images = train_images.reshape(-1, image_y, image_x, 1)

# Define CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_y, image_x, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')  # num_classes is the number of gesture classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save("gesture_cnn_model.h5")

print("Training completed and model saved.")
