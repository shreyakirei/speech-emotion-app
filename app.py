import streamlit as st
import librosa
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

# Emotion dictionary mapping predicted labels to emotion names
emotion_dict = {
    "0": "neutral",
    "1": "calm",
    "2": "happy",
    "3": "sad",
    "4": "angry",
    "5": "fearful",
    "6": "disgust",
    "7": "surprised"
}

# Apply custom CSS styles for light pink background and cute UI
st.markdown(
    """
    <style>
    /* Background color for whole app */
    .stApp {
        background-color: #ffebf0 !important;
    }

    /* Main text styling */
    body, .css-1d391kg, .css-1v3fvcr {
        color: #c94f7c !important;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }

    /* Style primary buttons */
    button[kind="primary"] {
        background-color: #fbcfe8 !important;
        color: #7a3d5d !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        border: 2px solid #f9a8d4 !important;
        padding: 10px 20px !important;
    }

    button[kind="primary"]:hover {
        background-color: #f9a8d4 !important;
    }

    /* Center title */
    .css-10trblm h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.title("🌸 Speech Emotion Detector { try this with your own audacity .wav file!!! ")

# File uploader widget
uploaded_file = st.file_uploader("💖 Upload a WAV audio file", type=["wav"])

# Function to extract MFCC features from audio
def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

# If a file is uploaded, process and predict emotion
if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    features = extract_features(uploaded_file)

    if features is not None:
        prediction = model.predict([features])[0]
        emotion = emotion_dict.get(str(prediction), "unknown")

        st.markdown(f"""
        <div style='text-align: center; margin-top: 30px;'>
            <h2>🎯 Detected Emotion:</h2>
            <h1 style='color: #db2777;'>💘 {emotion.upper()} 💘</h1>
        </div>
        """, unsafe_allow_html=True)
