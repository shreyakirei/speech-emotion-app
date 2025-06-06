import streamlit as st
import librosa
import numpy as np
import pickle

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

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

st.markdown(
    """
    <style>
    /* Background color */
    .stApp {
        background-color: #ffebf0 !important;
    }

    /* Force black color for all text elements */
    .stApp, 
    .stApp * {
        color: #000000 !important;
        font-family: 'Comic Sans MS', cursive, sans-serif !important;
    }

    /* Style buttons */
    button[kind="primary"] {
        background-color: #fbcfe8 !important;
        color: #4b0079 !important;
        border-radius: 12px !important;
        font-size: 16px !important;
        border: 2px solid #f9a8d4 !important;
        padding: 10px 20px !important;
    }

    button[kind="primary"]:hover {
        background-color: #f9a8d4 !important;
    }

    /* Center the title */
    .css-10trblm h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("‪‪❤︎‬ Speech Emotion Detector {try with your own audacity .wav file! ‪‪❤︎‬")

uploaded_file = st.file_uploader("💖 Upload a WAV audio file", type=["wav"])

def extract_features(file):
    try:
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    features = extract_features(uploaded_file)

    if features is not None:
        prediction = model.predict([features])[0]
        emotion = emotion_dict.get(str(prediction), "unknown")

        st.markdown(f"""
        <div style='text-align: center; margin-top: 30px;'>
            <h2>🎯 Detected Emotion:</h2>
            <h1>💘 {emotion.upper()} 💘</h1>
        </div>
        """, unsafe_allow_html=True)

