import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Cache the model to avoid reloading on each run
@st.cache_resource
def load_emotion_model():
    return load_model('model_weights.keras')

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessing function
def image_preprocessing(img):
    img = cv2.resize(img, (48, 48))  # Resize to 48x48
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Histogram equalization
    denoised = cv2.GaussianBlur(equalized, (5, 5), 0)  # Reduce noise
    normalized = denoised / 255.0  # Normalize
    normalized = np.expand_dims(normalized, axis=-1)  # Add channel dimension
    return normalized

# Load the model
model = load_emotion_model()

# Streamlit app
st.title("Facial Expression Recognition")
st.write("Upload an image of a face to predict the emotion.")

# Upload image
uploaded_img = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_img is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_img).convert('RGB')  # Ensure 3-channel RGB format
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Convert to numpy array
        image_np = np.array(image)

        # Preprocess the image
        preprocessed_image = image_preprocessing(image_np)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = model.predict(preprocessed_image)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        confidence_score = np.max(prediction)

        st.write(f"**Predicted Emotion:** {predicted_emotion}")
        st.write(f"**Confidence Score:** {confidence_score:.2f}")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
