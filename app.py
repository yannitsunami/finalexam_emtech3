import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model('gold_silver_model.h5')

st.title("🥇 Gold vs Silver Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128,128))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Gold" if prediction > 0.5 else "Silver"
    st.write(f"Prediction: **{label}**")
