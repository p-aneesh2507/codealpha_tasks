import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================
# 1. SETTINGS
# ============================================

DATASET_PATH = "dataset/archive/audio_speech_actors_01-24"
MODEL_PATH = "model/emotion_model.keras"

SAMPLE_RATE = 22050
MAX_PAD_LEN = 174


# ============================================
# 2. EMOTION MAPPING
# ============================================

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ============================================
# 3. MFCC FEATURE EXTRACTION
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
# 4. LOAD DATASET
# ============================================

features = []
labels = []

print("\nLoading audio files...\n")


for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    for filename in os.listdir(actor_path):

        if filename.endswith(".wav"):

            file_path = os.path.join(
                actor_path,
                filename
            )

            try:

                # RAVDESS filename example:
                # 03-01-05-01-02-01-12.wav
                #
                # Third value = emotion code

                emotion_code = filename.split("-")[2]

                emotion = emotion_map.get(
                    emotion_code
                )

                if emotion is None:
                    continue

                mfcc = extract_features(
                    file_path
                )

                features.append(mfcc)
                labels.append(emotion)

            except Exception as e:

                print(
                    f"Error processing {filename}: {e}"
                )


# ============================================
# 5. CONVERT DATA TO NUMPY
# ============================================

X = np.array(features)
y = np.array(labels)

print("\nDataset loaded successfully!")

print(
    "X shape:",
    X.shape
)

print(
    "Number of samples:",
    len(X)
)


# ============================================
# CHECK DATASET
# ============================================

if len(X) == 0:

    print("\nERROR: No audio files were found.")

    print(
        "Check DATASET_PATH in train.py."
    )

    exit()


# ============================================
# 6. ENCODE EMOTIONS
# ============================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(
    y
)

y_categorical = tf.keras.utils.to_categorical(
    y_encoded
)


print("\nEmotions:")

for index, emotion in enumerate(
    label_encoder.classes_
):

    print(
        index,
        "=",
        emotion
    )


# ============================================
# 7. TRAIN / TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y_categorical,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================
# 8. BUILD CNN MODEL
# ============================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(40, MAX_PAD_LEN)
    ),

    tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu"
    ),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Conv1D(
        filters=128,
        kernel_size=3,
        activation="relu"
    ),

    tf.keras.layers.MaxPooling1D(
        pool_size=2
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.3
    ),

    tf.keras.layers.Dense(
        len(label_encoder.classes_),
        activation="softmax"
    )
])


# ============================================
# 9. COMPILE MODEL
# ============================================

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]
)


print("\nModel Summary:\n")

model.summary()


# ============================================
# 10. TRAIN MODEL
# ============================================

print("\nTraining model...\n")


history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=30,

    batch_size=32,

    verbose=1
)


# ============================================
# 11. EVALUATE MODEL
# ============================================

test_loss, test_accuracy = model.evaluate(

    X_test,

    y_test,

    verbose=0
)


print("\n================================")
print("MODEL RESULTS")
print("================================")


print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)


# ============================================
# 12. CLASSIFICATION REPORT
# ============================================

predictions = model.predict(
    X_test
)


predicted_classes = np.argmax(
    predictions,
    axis=1
)


actual_classes = np.argmax(
    y_test,
    axis=1
)


print(
    "\nClassification Report:\n"
)


print(
    classification_report(

        actual_classes,

        predicted_classes,

        target_names=label_encoder.classes_
    )
)


# ============================================
# 13. CONFUSION MATRIX
# ============================================

cm = confusion_matrix(

    actual_classes,

    predicted_classes
)


plt.figure(
    figsize=(10, 8)
)


plt.imshow(cm)


plt.title(
    "Speech Emotion Recognition - Confusion Matrix"
)


plt.xlabel(
    "Predicted Emotion"
)


plt.ylabel(
    "Actual Emotion"
)


plt.xticks(

    range(
        len(label_encoder.classes_)
    ),

    label_encoder.classes_,

    rotation=45
)


plt.yticks(

    range(
        len(label_encoder.classes_)
    ),

    label_encoder.classes_
)


plt.colorbar()


plt.tight_layout()


plt.show()


# ============================================
# 14. ACCURACY GRAPH
# ============================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(

    history.history["accuracy"],

    label="Training Accuracy"
)


plt.plot(

    history.history["val_accuracy"],

    label="Validation Accuracy"
)


plt.title(
    "Training vs Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Accuracy"
)


plt.legend()


plt.show()


# ============================================
# 15. SAVE MODEL
# ============================================

os.makedirs(

    "model",

    exist_ok=True
)


model.save(
    MODEL_PATH
)


print(
    f"\nModel saved successfully at: {MODEL_PATH}"
)


print(
    "\nPROJECT TRAINING COMPLETED!"
)