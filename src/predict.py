import os
import sys
import joblib
import librosa
import numpy as np


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/emotion_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ============================================================
# 2. FEATURE EXTRACTION
# ============================================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_features = np.mean(
        mfcc,
        axis=1
    )

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    zcr_mean = np.mean(zcr)

    # RMS Energy
    rms = librosa.feature.rms(
        y=audio
    )

    rms_mean = np.mean(rms)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_mean = np.mean(
        spectral_centroid
    )

    # Combine all features
    features = np.concatenate([
        mfcc_features,
        [zcr_mean],
        [rms_mean],
        [spectral_mean]
    ])

    return features


# ============================================================
# 3. PREDICT EMOTION
# ============================================================

def predict_emotion(file_path):

    features = extract_features(
        file_path
    )

    features = features.reshape(
        1, -1
    )

    prediction = model.predict(
        features
    )

    emotion = label_encoder.inverse_transform(
        prediction
    )[0]

    return emotion


# ============================================================
# 4. MAIN PROGRAM
# ============================================================

if len(sys.argv) < 2:

    print("\nPlease provide an audio file.")
    print("\nExample:")
    print(
        "python src/predict.py "
        "test_audio/sample.wav"
    )

    sys.exit()


audio_file = sys.argv[1]


if not os.path.exists(audio_file):

    print(
        f"\nError: File not found: {audio_file}"
    )

    sys.exit()


print("=" * 60)
print("          SPEECH EMOTION PREDICTION")
print("=" * 60)

print("\nAudio File:")
print(audio_file)

print("\nAnalyzing audio...")

emotion = predict_emotion(
    audio_file
)

print("\n" + "=" * 60)

print("Predicted Emotion:")
print(
    emotion.upper()
)

print("=" * 60)