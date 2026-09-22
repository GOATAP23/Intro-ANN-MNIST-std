import os
import numpy as np
import PIL.ImageOps
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="MNIST Digit Predictor", page_icon="🔢", layout="centered")

MODEL_PATH = "67102010176_mnist_model.keras"

@st.cache_resource
def load_model(path: str):
    """Cache the model in memory to avoid reloading on every interaction."""
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

st.title("🔢 MNIST Digit Predictor")
st.write("Upload an image of a handwritten digit to get a model prediction.")

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"Model file `{MODEL_PATH}` not found. Please verify the model file path.")
else:
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("L")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)

            # Option to invert colors (MNIST models expect white digits on a black background)
            invert_colors = st.checkbox(
                "Invert colors (check if digit is dark on white paper)", value=True
            )

            if invert_colors:
                image = PIL.ImageOps.invert(image)

            # Preprocessing: Resize to 28x28 and normalize to [0, 1]
            img_resized = image.resize((28, 28))
            img_array = np.array(img_resized, dtype=np.float32) / 255.0

            # Match model input shape: handles either (1, 28, 28) or (1, 28, 28, 1)
            expected_shape = model.input_shape
            if len(expected_shape) == 4:
                img_input = np.expand_dims(img_array, axis=(0, -1))
            else:
                img_input = np.expand_dims(img_array, axis=0)

            # Predict
            predictions = model.predict(img_input, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(predictions[predicted_digit]) * 100

            with col2:
                st.subheader("Prediction")
                st.metric(label="Predicted Digit", value=str(predicted_digit))
                st.metric(label="Confidence", value=f"{confidence:.2f}%")

            # Probability Breakdown
            st.markdown("---")
            st.subheader("Class Probabilities")
            st.bar_chart(predictions)

        except Exception as e:
            st.error(f"An error occurred while processing the image: {e}")
