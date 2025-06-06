import streamlit as st
import librosa
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Define emotions
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

# Set page config and styling
st.set_page_config(page_title="🎀 Speech Emotion App", page_icon="🎧", layout="centered")

st.markdown("""
<style>
body {
    background-color: #ffe4e6;
    font-family: 'Comic Sans MS', cursive, sans-serif;
    color: #c94f7c;
}
h1, h2, h3 {
    color: #c94f7c;
}
.stButton>button {
    background-color: #fbcfe8;
    color: #7a3d5d;
    border: 2px solid #f9a8d4;
    border-radius: 12px;
    font-size: 16px;
    padding: 10px 20px;
}
.stButton>button:hover {
    background-color: #f9a8d4;
}
</style>
""", unsafe_allow_html=True)

st.title("🌸 Cute Speech Emotion Detector")

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
            <h1 style='color: #db2777;'>💘 {emotion.upper()} 💘</h1>
        </div>
        """, unsafe_allow_html=True)
