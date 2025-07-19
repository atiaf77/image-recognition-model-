# image-recognition-model-
Train an image recognition model using Teachable Machine by Google Files
keras_model.h5
The exported TensorFlow Keras model file from Teachable Machine.

predict.py
Python script to load the trained model, accept an image path as input, and output the predicted class with confidence.

test.jpg
Sample image used to test the prediction script.

README.md
This file, explaining the project setup and usage.

Setup Instructions
Install required Python packages:

bash

pip install tensorflow pillow numpy
Place the trained model file (keras_model.h5), the predict.py script, and your test image (e.g., test.jpg) in the same directory.

Usage
Run the prediction script from the command line with the path to the image you want to classify:

bash

python predict.py test.jpg
Expected output example:

makefile

Prediction: Cat (95.67%)
How to Train Your Own Model
Go to Teachable Machine.

Choose Image Project → Standard Image Model.

Create at least two classes and upload images for each.

Train the model.

Export the model by selecting TensorFlow and download the exported model files.

Replace the keras_model.h5 in this project with your downloaded model file.

Notes
The input image should be in JPG or PNG format.

The image will be resized automatically to 224x224 pixels as required by the model.

Make sure the class names in predict.py match the classes you trained in Teachable Machine.

Python Script (predict.py)
python

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys


model = tf.keras.models.load_model('keras_model.h5')

class_names = ['disease,healthy']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224)) 
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(img_path):
    img = prepare_image(img_path)
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx] * 100
    print(f"Prediction: {class_names[class_idx]} ({confidence:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict(sys.argv[1])

This project demonstrates training an image recognition model using Google Teachable Machine with at least two classes, exporting the trained model in TensorFlow Keras format, and writing a Python script to load the model and predict the class of an input image.

