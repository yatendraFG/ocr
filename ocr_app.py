import streamlit as st
import cv2
import pytesseract
import os
import numpy as np
from PIL import Image
import joblib
import tensorflow as tf

# File paths – adjust if yours are different
base_path = r'G:\MyDrive\Project-10'
model_path = r'G:\My Drive\Project-10\best_model.keras' # Path to your saved .keras model
vectorizer_path = r'G:\My Drive\Project-10\tfidf_vectorizer.pkl'  # Path to your saved vectorizer
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the trained Keras model
model = tf.keras.models.load_model(model_path)

# Load the saved TF-IDF vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = joblib.load(f)

# Streamlit UI
st.title("OCR Text Extraction & Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format and grayscale
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR: extract text from the image
    extracted_text = pytesseract.image_to_string(gray)
    st.write("### Extracted Text:")
    st.write(extracted_text)

    if extracted_text.strip():  # Only predict if text is not empty
        # TF-IDF vectorization
        text_features = vectorizer.transform([extracted_text]).toarray()
        text_features = text_features.reshape(text_features.shape[0], 1, text_features.shape[1])

        # Predict with the model
        prediction = model.predict(text_features)
        
        st.write("### Model Prediction:")
        st.write(prediction.tolist())
    else:
        st.warning("No text was extracted from the image.")
