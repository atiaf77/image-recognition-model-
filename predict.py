import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import sys

# Load the trained model
model = load_model("keras_model.h5")

# Load class labels from labels.txt
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict(img_path):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    index = np.argmax(predictions[0])
    confidence = predictions[0][index]
    label = class_names[index]
    print(f"Prediction: {label} ({confidence:.2%})")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict(sys.argv[1])
