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

st.set_page_config(page_title="🎤 Speech Emotion Recognizer", layout="centered")
st.title("🎧 Speech Emotion Detection App")

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
