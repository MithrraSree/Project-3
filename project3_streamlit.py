import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
@st.cache_resource
def load_trained_model():
    model = load_model("C:/Users/Mithrra Sree/Downloads/vgg_model.h5")  
    return model

model = load_trained_model()

# App title
st.title("Chest X-ray TB Classifier")
st.write("Upload a chest X-ray image to check if it shows signs of **Tuberculosis**.")

# Upload image
uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-ray', use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize (must match training)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    raw_pred = model.predict(img_array)
    st.write("🔍 Raw prediction output:", raw_pred)

    confidence = raw_pred[0][0]

    # Show probability instead of binary decision
    if confidence >= 0.75:
        result = "Tuberculosis"
    elif confidence >= 0.5:
        result = "Possibly Tuberculosis"
    else:
        result = "Normal"

    # Show result
    st.subheader("Prediction:")
    st.write(f" **{result}** (Confidence: {confidence:.2f})")

