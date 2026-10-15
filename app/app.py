import os
import tempfile

import joblib
import librosa
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import librosa.display


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = "models/emotion_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_file):

    audio, sample_rate = librosa.load(
        audio_file,
        sr=None
    )

    # MFCC
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

    # Combine features
    features = np.concatenate([
        mfcc_features,
        [zcr_mean],
        [rms_mean],
        [spectral_mean]
    ])

    return features, audio, sample_rate


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🎤 Speech Emotion Recognition")

st.write(
    "Upload a speech audio file and the machine learning "
    "model will predict the emotion expressed in the speech."
)

st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a WAV audio file",
    type=["wav"]
)


# ============================================================
# PROCESS AUDIO
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded file: {uploaded_file.name}"
    )

    # Play audio
    st.audio(
        uploaded_file,
        format="audio/wav"
    )

    # Save temporary file
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as temp_file:

        temp_file.write(
            uploaded_file.read()
        )

        temp_audio_path = temp_file.name

    try:

        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        features, audio, sample_rate = extract_features(
            temp_audio_path
        )

        features = features.reshape(
            1, -1
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        prediction = model.predict(
            features
        )

        emotion = label_encoder.inverse_transform(
            prediction
        )[0]


        # ====================================================
        # EMOTION PROBABILITIES
        # ====================================================

        probabilities = model.predict_proba(
            features
        )[0]

        emotion_names = label_encoder.classes_

        probability_data = dict(
            zip(
                emotion_names,
                probabilities
            )
        )

        confidence = np.max(
            probabilities
        )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.subheader("🎯 Predicted Emotion")

        st.success(
            emotion.upper()
        )

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # EMOTION PROBABILITIES
        # ====================================================

        st.subheader("📊 Emotion Probabilities")

        sorted_probabilities = dict(
            sorted(
                probability_data.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        for emotion_name, probability in sorted_probabilities.items():

            st.write(
                f"**{emotion_name.capitalize()}**"
            )

            st.progress(
                float(probability)
            )

            st.write(
                f"{probability * 100:.2f}%"
            )


        # ====================================================
        # AUDIO INFORMATION
        # ====================================================

        st.subheader("📊 Audio Information")

        duration = librosa.get_duration(
            y=audio,
            sr=sample_rate
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Sampling Rate",
                f"{sample_rate} Hz"
            )

        with col2:
            st.metric(
                "Duration",
                f"{duration:.2f} seconds"
            )

        with col3:
            st.metric(
                "Number of Samples",
                len(audio)
            )


        # ====================================================
        # WAVEFORM
        # ====================================================

        st.subheader("〰️ Speech Waveform")

        fig, ax = plt.subplots(
            figsize=(12, 4)
        )

        librosa.display.waveshow(
            audio,
            sr=sample_rate,
            ax=ax
        )

        ax.set_title(
            "Speech Waveform"
        )

        ax.set_xlabel(
            "Time (seconds)"
        )

        ax.set_ylabel(
            "Amplitude"
        )

        st.pyplot(fig)

        plt.close(fig)


        # ====================================================
        # MFCC
        # ====================================================

        st.subheader("📈 MFCC Features")

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        fig, ax = plt.subplots(
            figsize=(12, 5)
        )

        img = librosa.display.specshow(
            mfcc,
            sr=sample_rate,
            x_axis="time",
            ax=ax
        )

        fig.colorbar(
            img,
            ax=ax
        )

        ax.set_title(
            "MFCC Features"
        )

        st.pyplot(fig)

        plt.close(fig)


    except Exception as e:

        st.error(
            f"Error processing audio: {e}"
        )


    finally:

        if os.path.exists(
            temp_audio_path
        ):

            os.remove(
                temp_audio_path
            )