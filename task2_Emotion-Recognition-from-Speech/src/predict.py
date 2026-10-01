import os
import numpy as np
import librosa
import tensorflow as tf


# ============================================
# 1. SETTINGS
# ============================================

MODEL_PATH = "model/emotion_model.keras"

SAMPLE_RATE = 22050
MAX_PAD_LEN = 174


# ============================================
# 2. EMOTION LABELS
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
# 3. LOAD TRAINED MODEL
# ============================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================
# 4. FEATURE EXTRACTION
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
# 5. GET AUDIO FILE
# ============================================

audio_path = input(
    "\nEnter path of your .wav audio file: "
).strip()


# Remove quotes if user pastes a path like:
# "audio/test.wav"

audio_path = audio_path.strip('"')


# ============================================
# 6. CHECK FILE
# ============================================

if not os.path.exists(audio_path):

    print(
        "\nERROR: Audio file not found!"
    )

    print(
        "Please check the file path."
    )

    exit()


if not audio_path.lower().endswith(".wav"):

    print(
        "\nERROR: Please provide a .wav file."
    )

    exit()


# ============================================
# 7. EXTRACT MFCC
# ============================================

print(
    "\nExtracting audio features..."
)

features = extract_features(
    audio_path
)


# ============================================
# 8. PREPARE INPUT
# ============================================

features = np.expand_dims(
    features,
    axis=0
)


# ============================================
# 9. PREDICT EMOTION
# ============================================

print(
    "Predicting emotion..."
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


# ============================================
# 10. DISPLAY RESULT
# ============================================

print("\n================================")
print("       EMOTION PREDICTION")
print("================================")

print(
    f"Predicted Emotion : {predicted_emotion}"
)

print(
    f"Confidence        : {confidence:.2f}%"
)

print("================================")