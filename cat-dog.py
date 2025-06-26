import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your model file (make sure it's in the same folder as this script)
model = tf.keras.models.load_model("cat_dog_model.h5")

# Define prediction function
def classify_image(img):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)
    prediction = model.predict(img_array)[0][0]
    return "Dog 🐶" if prediction > 0.5 else "Cat 🐱"

# Gradio UI
app = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Cat or Dog Classifier",
    description="Upload an image to find out if it's a cat or a dog!"
)

app.launch()
