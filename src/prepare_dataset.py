import os
import librosa
import numpy as np
import pandas as pd


# ==================================================
# RAVDESS DATASET LOCATION
# ==================================================

DATASET_PATH = (
    "dataset/RAVDESS/"
    "Audio_Speech_Actors_01-24"
)


# ==================================================
# EMOTION MAPPING
# ==================================================

emotion_mapping = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ==================================================
# FEATURE EXTRACTION FUNCTION
# ==================================================

def extract_features(file_path):

    audio, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    # ----------------------------------------------
    # MFCC
    # ----------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_features = np.mean(
        mfcc,
        axis=1
    )

    # ----------------------------------------------
    # Zero Crossing Rate
    # ----------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        audio
    )

    zcr_mean = np.mean(zcr)

    # ----------------------------------------------
    # RMS Energy
    # ----------------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    rms_mean = np.mean(rms)

    # ----------------------------------------------
    # Spectral Centroid
    # ----------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_mean = np.mean(
        spectral_centroid
    )

    # ----------------------------------------------
    # Combine all features
    # ----------------------------------------------

    features = np.concatenate([
        mfcc_features,
        [zcr_mean],
        [rms_mean],
        [spectral_mean]
    ])

    return features


# ==================================================
# PROCESS ALL AUDIO FILES
# ==================================================

all_features = []
all_labels = []
all_files = []


print("=" * 60)
print("      RAVDESS FEATURE EXTRACTION")
print("=" * 60)


# Go through Actor_01 to Actor_24

for actor_folder in sorted(os.listdir(DATASET_PATH)):

    actor_path = os.path.join(
        DATASET_PATH,
        actor_folder
    )

    # Make sure it is a folder
    if not os.path.isdir(actor_path):
        continue

    print(f"\nProcessing {actor_folder}...")

    # Find all WAV files

    for filename in sorted(os.listdir(actor_path)):

        if not filename.endswith(".wav"):
            continue

        file_path = os.path.join(
            actor_path,
            filename
        )

        try:

            # --------------------------------------
            # Extract emotion code from filename
            # --------------------------------------

            parts = filename.split("-")

            emotion_code = parts[2]

            emotion = emotion_mapping[
                emotion_code
            ]

            # --------------------------------------
            # Extract audio features
            # --------------------------------------

            features = extract_features(
                file_path
            )

            # --------------------------------------
            # Store results
            # --------------------------------------

            all_features.append(features)
            all_labels.append(emotion)
            all_files.append(file_path)

        except Exception as e:

            print(
                f"Error processing {filename}: {e}"
            )


# ==================================================
# CREATE DATAFRAME
# ==================================================

feature_columns = [
    f"mfcc_{i + 1}"
    for i in range(40)
]

feature_columns += [
    "zcr",
    "rms",
    "spectral_centroid"
]


df = pd.DataFrame(
    all_features,
    columns=feature_columns
)


df["emotion"] = all_labels
df["file"] = all_files


# ==================================================
# SAVE CSV
# ==================================================

os.makedirs(
    "data",
    exist_ok=True
)


output_file = "data/features.csv"


df.to_csv(
    output_file,
    index=False
)


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION COMPLETED")
print("=" * 60)

print(f"Total audio files processed : {len(df)}")

print(
    f"Number of features          : "
    f"{len(feature_columns)}"
)

print(f"CSV file saved at           : {output_file}")


print("\nEmotion distribution:")
print(
    df["emotion"].value_counts()
)

print("=" * 60)