import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv("data/features.csv")


print("=" * 60)
print("        SPEECH EMOTION DATASET ANALYSIS")
print("=" * 60)


# ==================================================
# BASIC INFORMATION
# ==================================================

print("\nDataset Shape:")
print(df.shape)


print("\nFirst 5 Rows:")
print(df.head())


print("\nDataset Information:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum().sum())


# ==================================================
# EMOTION DISTRIBUTION
# ==================================================

print("\nEmotion Distribution:")
print(df["emotion"].value_counts())


plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="emotion"
)

plt.title("Distribution of Speech Emotions")
plt.xlabel("Emotion")
plt.ylabel("Number of Audio Files")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "visualizations/emotion_distribution.png"
)

plt.show()


# ==================================================
# FEATURE STATISTICS
# ==================================================

print("\nFeature Statistics:")

print(
    df[
        [
            "zcr",
            "rms",
            "spectral_centroid"
        ]
    ].describe()
)


# ==================================================
# AVERAGE FEATURES BY EMOTION
# ==================================================

emotion_features = df.groupby(
    "emotion"
)[
    [
        "zcr",
        "rms",
        "spectral_centroid"
    ]
].mean()


print("\nAverage Audio Features by Emotion:")

print(emotion_features)


# ==================================================
# FEATURE COMPARISON
# ==================================================

emotion_features.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Average Audio Features by Emotion"
)

plt.xlabel("Emotion")

plt.ylabel("Average Value")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "visualizations/feature_comparison.png"
)

plt.show()


# ==================================================
# CORRELATION HEATMAP
# ==================================================

correlation_features = df[
    [
        "mfcc_1",
        "mfcc_2",
        "mfcc_3",
        "mfcc_4",
        "mfcc_5",
        "zcr",
        "rms",
        "spectral_centroid"
    ]
]


plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_features.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Audio Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png"
)

plt.show()


# ==================================================
# MFCC DISTRIBUTION
# ==================================================

plt.figure(figsize=(12, 6))

sns.boxplot(
    data=df,
    x="emotion",
    y="mfcc_1"
)

plt.title(
    "MFCC-1 Distribution Across Emotions"
)

plt.xlabel("Emotion")

plt.ylabel("MFCC-1")

plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "visualizations/mfcc_distribution.png"
)

plt.show()


# ==================================================
# COMPLETED
# ==================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "Visualizations saved in the visualizations folder."
)