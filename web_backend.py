
import logging
import os
import tempfile
from pathlib import Path

import joblib
import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "emotion_model.pkl"
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

EXPECTED_FEATURES = 43
MAX_UPLOAD_MB = 25
MAX_AUDIO_SECONDS = 120

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)


# =========================================================
# 2. LOAD THE EXISTING MODEL
# =========================================================

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not ENCODER_PATH.is_file():
    raise FileNotFoundError(f"Label encoder not found: {ENCODER_PATH}")

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)

model_classes = getattr(model, "classes_", None)
expected_model_features = getattr(model, "n_features_in_", None)

app.logger.info("EchoMind model loaded: %s", type(model).__name__)
app.logger.info("Emotion labels: %s", label_encoder.classes_)
app.logger.info(
    "Model expected feature count: %s",
    expected_model_features
)

# Catch a common training/backend mismatch immediately.
if (
    expected_model_features is not None
    and expected_model_features != EXPECTED_FEATURES
):
    raise ValueError(
        f"The saved model expects {expected_model_features} features, "
        f"but this backend extracts {EXPECTED_FEATURES}. "
        "Check the original training code before using this backend."
    )


# =========================================================
# 3. AUDIO FEATURE EXTRACTION
# =========================================================

def extract_features(file_path):
    """
    Extract 43 features:
    40 mean MFCCs + mean ZCR + mean RMS + mean spectral centroid.

    IMPORTANT:
    This must match the feature extraction used during training.
    """

    audio, sample_rate = librosa.load(
        str(file_path),
        sr=None,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("The uploaded audio file is empty.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("The audio contains invalid numeric values.")

    duration = len(audio) / sample_rate

    if duration > MAX_AUDIO_SECONDS:
        raise ValueError(
            f"Audio is too long. Please upload a recording "
            f"of {MAX_AUDIO_SECONDS} seconds or less."
        )

    # Detect recordings that contain no meaningful audio.
    peak = float(np.max(np.abs(audio)))
    rms_level = float(np.sqrt(np.mean(np.square(audio))))

    if peak < 1e-5 or rms_level < 1e-5:
        raise ValueError(
            "The recording is silent or too quiet. "
            "Please upload a clearer speech recording."
        )

    # Feature 1-40: mean MFCC coefficients.
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )
    mfcc_features = np.mean(mfcc, axis=1)

    # Feature 41: mean zero-crossing rate.
    zcr = float(np.mean(
        librosa.feature.zero_crossing_rate(audio)
    ))

    # Feature 42: mean root mean square energy.
    rms = float(np.mean(
        librosa.feature.rms(y=audio)
    ))

    # Feature 43: mean spectral centroid.
    spectral_centroid = float(np.mean(
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sample_rate
        )
    ))

    features = np.concatenate([
        mfcc_features,
        [zcr, rms, spectral_centroid]
    ]).astype(np.float64)

    if features.shape != (EXPECTED_FEATURES,):
        raise ValueError(
            f"Feature extraction produced {features.size} features; "
            f"{EXPECTED_FEATURES} were expected."
        )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Audio feature extraction produced invalid values."
        )

    return features, audio, sample_rate


# =========================================================
# 4. MAP MODEL CLASSES TO EMOTION LABELS
# =========================================================

def decode_emotion(class_value):
    """
    Support models trained on LabelEncoder integer classes
    and models whose classes are already emotion strings.
    """

    encoder_classes = np.asarray(label_encoder.classes_)

    # Models trained using LabelEncoder usually predict 0, 1, 2...
    try:
        numeric_value = float(class_value)

        if (
            np.isfinite(numeric_value)
            and numeric_value.is_integer()
        ):
            class_index = int(numeric_value)

            if 0 <= class_index < len(encoder_classes):
                return str(
                    label_encoder.inverse_transform(
                        np.array([class_index], dtype=int)
                    )[0]
                )
    except (TypeError, ValueError, OverflowError):
        pass

    # Also support models trained directly on string labels.
    text_value = str(class_value)

    if text_value in {str(value) for value in encoder_classes}:
        return text_value

    raise ValueError(
        f"Model class {class_value!r} could not be matched "
        "to the saved emotion labels. Check the training labels "
        "and label encoder."
    )


# =========================================================
# 5. MAIN WEBSITE
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# 6. EMOTION PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Validate the uploaded file.
    if "audio" not in request.files:
        return jsonify({
            "error": "No audio file was received. Please select a WAV file."
        }), 400

    uploaded_file = request.files["audio"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "Please select an audio file."
        }), 400

    if Path(uploaded_file.filename).suffix.lower() != ".wav":
        return jsonify({
            "error": "Unsupported file type. Please upload a WAV file."
        }), 400

    temp_path = None

    try:
        # Save to a temporary file; never trust the original filename.
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_path = temp_file.name
            uploaded_file.save(temp_file)

        if os.path.getsize(temp_path) == 0:
            return jsonify({
                "error": "The uploaded file is empty."
            }), 400

        # Extract the same features used by the current model.
        features, audio, sample_rate = extract_features(temp_path)
        model_input = features.reshape(1, -1)

        # Predict the model's class.
        prediction = model.predict(model_input)
        predicted_class = prediction[0]
        emotion = decode_emotion(predicted_class)

        # Calculate class probabilities, if supported.
        probability_data = {}
        confidence = 0.0

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(
                model.predict_proba(model_input)[0],
                dtype=float
            )

            classes = (
                np.asarray(model_classes)
                if model_classes is not None
                else np.arange(len(probabilities))
            )

            if len(classes) != len(probabilities):
                raise ValueError(
                    "The model's classes and probability outputs "
                    "do not match."
                )

            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0)
                or not np.isclose(np.sum(probabilities), 1.0, atol=1e-3)
            ):
                raise ValueError(
                    "The model returned invalid class probabilities."
                )

            for class_value, probability in zip(classes, probabilities):
                label = decode_emotion(class_value)
                probability_data[label] = round(
                    float(probability) * 100, 2
                )

            # Confidence is the probability of the predicted class,
            # rather than blindly assuming the array's largest value.
            predicted_label = decode_emotion(predicted_class)
            confidence = float(
                probabilities[
                    next(
                        i for i, class_value in enumerate(classes)
                        if decode_emotion(class_value) == predicted_label
                    )
                ]
            ) * 100

        else:
            app.logger.warning(
                "The current model does not support predict_proba(). "
                "A calibrated confidence score is unavailable."
            )

        duration = len(audio) / sample_rate

        app.logger.info(
            "Prediction completed: emotion=%s, confidence=%.2f%%, "
            "duration=%.2fs, sample_rate=%s",
            emotion,
            confidence,
            duration,
            sample_rate
        )

        # Keep the response keys compatible with script.js.
        return jsonify({
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "probabilities": probability_data,
            "duration": round(duration, 2),
            "sample_rate": int(sample_rate),
            "samples": int(len(audio))
        })

    except ValueError as error:
        app.logger.warning("Audio validation/prediction issue: %s", error)
        return jsonify({"error": str(error)}), 400

    except Exception:
        app.logger.exception("Unexpected audio prediction failure")
        return jsonify({
            "error": (
                "The audio could not be analyzed. Please try a valid, "
                "clear speech WAV file. Check the VS Code terminal "
                "for technical details."
            )
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                app.logger.warning(
                    "Could not remove temporary audio file: %s",
                    temp_path
                )


# =========================================================
# 7. UPLOAD SIZE ERROR
# =========================================================

@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(error):
    return jsonify({
        "error": (
            f"The audio file is too large. "
            f"Please upload a file smaller than {MAX_UPLOAD_MB} MB."
        )
    }), 413


# =========================================================
# 8. START FLASK
# =========================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
