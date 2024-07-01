from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import RandomHeight, RandomWidth, RandomFlip, RandomZoom, RandomRotation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the model with custom objects
custom_objects = {
    'RandomHeight': RandomHeight,
    'RandomWidth': RandomWidth,
    'RandomFlip': RandomFlip,
    'RandomZoom': RandomZoom,
    'RandomRotation': RandomRotation
}

model = tf.keras.models.load_model('animal_model.h5', custom_objects=custom_objects)

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = image.load_img(image_path, target_size=target_size)
    # Convert the image to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

# Prediction function
def predict_image(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['Butterfly', 'Cat', 'Chicken', 'Cow', 'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider', 'Squirrel']
    return class_labels[predicted_class[0]]

# Model function wrapper
def your_model_function(image_path):
    return predict_image(image_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result = your_model_function(filepath)
        return jsonify({'result': result})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, port=5001)
