import streamlit as st
import librosa
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# Set page title and layout
st.set_page_config(page_title="🎤 Speech Emotion Recognizer", layout="centered")

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* Background color for the whole app */
    .stApp {
        background-color: #ffe4e6 !important;  /* very light pink */
    }

    /* Make all text black and use a cute font */
    .stApp, .stApp * {
        color: #000000 !important;
        font-family: 'Comic Sans MS', cursive, sans-serif !important;
    }

    /* Style the file uploader button */
    div[role="button"] > label[for^="file"] {
        background-color: #f9c1d9 !important;  /* light pink */
        color: #3b0050 !important;  /* dark purple for contrast */
        border-radius: 12px !important;
        padding: 12px 20px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: 2px solid #f48fb1 !important;
        cursor: pointer !important;
        user-select: none !important;
        transition: background-color 0.3s ease !important;
    }

    /* Hover effect for uploader button */
    div[role="button"] > label[for^="file"]:hover {
        background-color: #f48fb1 !important;
        color: #1a0033 !important;
    }

    /* Center title text */
    .css-10trblm h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("‬Speech Emotion Detection App ‪‪❤︎🎧")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    features = extract_features(uploaded_file)

    if features is not None:
        prediction = model.predict([features])[0]
        st.write(f"Raw model prediction: {prediction} (type: {type(prediction)})")

        # Since prediction is a string label, use it directly
        emotion = prediction  
        st.success(f"🎯 Detected Emotion: **{emotion.upper()}**")
