import os
import numpy as np
import librosa
import tensorflow as tf
import streamlit as st


# ============================================
# SETTINGS
# ============================================

MODEL_PATH = "model/emotion_model.keras"

SAMPLE_RATE = 22050
MAX_PAD_LEN = 174


# ============================================
# EMOTION LABELS
# ============================================

emotion_labels = [
    "angry",
    "calm",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]


# ============================================
# PAGE SETTINGS
# ============================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered"
)


# ============================================
# TITLE
# ============================================

st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload a WAV audio file and the AI model "
    "will predict the emotion in the speech."
)


# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ============================================
# FEATURE EXTRACTION
# ============================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    if mfcc.shape[1] < MAX_PAD_LEN:

        pad_width = MAX_PAD_LEN - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :MAX_PAD_LEN]

    return mfcc


# ============================================
# FILE UPLOADER
# ============================================

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)


# ============================================
# PREDICTION
# ============================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    temp_file = "temp_audio.wav"

    with open(
        temp_file,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )


    if st.button("🔍 Predict Emotion"):

        with st.spinner(
            "Analyzing speech..."
        ):

            features = extract_features(
                temp_file
            )

            features = np.expand_dims(
                features,
                axis=0
            )

            prediction = model.predict(
                features,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            predicted_emotion = emotion_labels[
                predicted_index
            ]

            confidence = prediction[0][
                predicted_index
            ] * 100


        st.success(
            f"Predicted Emotion: {predicted_emotion.upper()}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


        # ====================================
        # ALL EMOTION PROBABILITIES
        # ====================================

        st.subheader(
            "Emotion Probabilities"
        )

        for i, emotion in enumerate(
            emotion_labels
        ):

            probability = (
                prediction[0][i] * 100
            )

            st.write(
                f"{emotion.capitalize()}: "
                f"{probability:.2f}%"
            )

            st.progress(
                float(prediction[0][i])
            )


    # Remove temporary file

    if os.path.exists(temp_file):

        os.remove(temp_file)