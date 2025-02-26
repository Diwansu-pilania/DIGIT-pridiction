# DIGIT-pridiction

📌 Handwritten Digit Recognition using CNN

🚀 This project implements a Convolutional Neural Network (CNN) to classify handwritten digits using the MNIST dataset. The model achieves high accuracy and serves as an introduction to deep learning in computer vision.

📖 Table of Contents
Introduction
Dataset
Installation
Model Architecture
Training & Evaluation
Results
Usage
Future Improvements
Contributors
📝 Introduction
The goal of this project is to train a deep learning model to recognize handwritten digits (0-9) using the MNIST dataset. The dataset consists of 60,000 training images and 10,000 test images, each being a 28x28 grayscale image.

✨ Key Features:
✅ Preprocessing using normalization and reshaping
✅ CNN model with multiple convolutional and dense layers
✅ Trained using TensorFlow/Keras for efficient learning
✅ Achieved high accuracy (~99%) on the test dataset

📊 Dataset
The MNIST dataset is a collection of handwritten digits widely used for training various image processing systems.

60,000 training images
10,000 testing images
Images are 28x28 pixels, grayscale
You can load the dataset using TensorFlow:

python
Copy
Edit
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
⚙️ Installation
To run this project, install the required dependencies using:

bash
Copy
Edit
pip install tensorflow numpy matplotlib
If you're using Google Colab, you can install additional dependencies with:

bash
Copy
Edit
!pip install torch torchvision matplotlib
🛠️ Model Architecture
The model consists of:

Convolutional Layers (Extract features from the images)
MaxPooling Layers (Reduce spatial dimensions)
Flatten Layer (Convert into a 1D array)
Fully Connected (Dense) Layers (Classify digits)
Softmax Activation (Output probability for each digit)
Model Summary:
python
Copy
Edit
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
📈 Training & Evaluation
The model is compiled with:
✅ Optimizer: Adam
✅ Loss Function: Sparse Categorical Crossentropy
✅ Metrics: Accuracy

python
Copy
Edit
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
After training, evaluate the model:

python
Copy
Edit
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')
📊 Results
✅ Training Accuracy: ~99%
✅ Test Accuracy: ~98-99%

A sample prediction:

python
Copy
Edit
import numpy as np
import matplotlib.pyplot as plt

image_index = 12  # Select an image
plt.imshow(x_test[image_index].reshape(28,28), cmap='gray')
pred = model.predict(x_test[image_index].reshape(1,28,28,1))
print(f'Predicted Digit: {np.argmax(pred)}')
🚀 Usage
To run the project, execute:

bash
Copy
Edit
python main.py
Or in Jupyter Notebook:

python
Copy
Edit
!jupyter notebook
For Google Colab, upload the notebook and run the cells.


🔮 Future Improvements
Data Augmentation to improve model generalization
Hyperparameter Tuning for better accuracy
Transfer Learning using pre-trained models
Deploying the Model as a web app using Flask or FastAPi


👨‍💻 Contributors
Diwansu

