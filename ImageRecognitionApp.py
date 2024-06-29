import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('animal_model.h5')

# Function to preprocess the image
def preprocess_image(image_path, target_size=(255, 255)):
    # Load the image
    img = image.load_img(image_path, target_size=target_size)
    # Convert the image to array
    img_array = image.img_to_array(img)
    # Expand dimensions to match the shape (1, 255, 255, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Normalize the image
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array

#Predictio function
def predict_image(image_path):
    target_size = (224, 224)  # or the size your model expects
    img = preprocess_image(image_path, target_size)
    
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = ['Butterfly', 'Cat', 'Chicken', 'Cow', 'Dog', 'Elephant', 'Hourse' ,'Sheep','Spider' ,'Squirrel']
    return class_labels[predicted_class[0]]


# Function to open a file dialog and select an image
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((255, 255))
        img = ImageTk.PhotoImage(img)
        panel.configure(image=img)
        panel.image = img
        label.config(text="Classifying...")
        label.update()
        prediction = predict_image(file_path)
        label.config(text=f"Prediction: {prediction}")

# Set up the GUI
root = tk.Tk()
root.title("Animal Image Classifier")

panel = Label(root)
panel.pack()

button = tk.Button(root, text="Select an image", command=open_file)
button.pack()

label = Label(root, text="Select an image to classify", font=("Helvetica", 16))
label.pack()

root.mainloop()
