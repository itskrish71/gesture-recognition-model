import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Set the path to your dataset
dataset_path = r'C:\Users\Krish\Desktop\WIndows\behaviour recognition\datasets'
categories = ['namaste-india', 'bowing-japan', 'handshake-western', 'mano-Philippines', 'righthandonheart-middleeast', 'tounguestickingout-tibet']  # Gesture categories

# Image size for resizing
img_size = 128


# Function to load and preprocess images from the dataset
def load_and_preprocess_data():
    data = []
    labels = []

    for category in categories:
        path = os.path.join(dataset_path, category)
        class_num = categories.index(category)  # Label the categories

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                resized_img = cv2.resize(img_array, (img_size, img_size))  # Resize image
                data.append(resized_img)
                labels.append(class_num)
            except Exception as e:
                pass

    data = np.array(data).reshape(-1, img_size, img_size, 1)  # Reshape for TensorFlow (Add a channel dimension)
    data = data / 255.0  # Normalize pixel values between 0 and 1

    return np.array(data), np.array(labels)


# Load data and split into train and test sets
data, labels = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build the convolutional neural network (CNN) model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))  # Output layer for 6 categories (namaste, bowing, handshake)

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Mapping labels to cultural responses
gesture_map = {0: ("🙏 namaste", "आपका भारत में स्वागत हैं"),
               1: ("🙇 Bowing", "こんにちは、元気ですか"),
               2: ("🤝 Handshake", "hello!!, how you doing ?"),
               3: ("🫂 mano", "Kumusta/cómo está ?"),
               4: ("🤝 righthandonheart", "مرحبا كيف حالك ?"),
               5: ("🤝 tounguestickingout", "Tashi Delek ?")}


# Function to preprocess a video frame for the model
def preprocess_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    resized_frame = cv2.resize(gray_frame, (img_size, img_size))  # Resize to model input size
    normalized_frame = resized_frame.reshape(-1, img_size, img_size, 1) / 255.0  # Normalize and reshape
    return normalized_frame


# Function to predict gesture from a frame
def predict_gesture(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    gesture_class = np.argmax(prediction)  # Get class with the highest probability
    response, culture = gesture_map[gesture_class]
    return response, culture


# Real-time video capture using OpenCV
cap = cv2.VideoCapture(0)  # 0 indicates the default camera

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame")
        break

    # Predict the gesture for the current frame
    gesture, culture = predict_gesture(frame)

    # Display the gesture and cultural information on the frame
    cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Culture: {culture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the resulting frame with the gesture and culture
    cv2.imshow('Cultural Gesture Recognition', frame)

    # Press 'q' to exit the video feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()
