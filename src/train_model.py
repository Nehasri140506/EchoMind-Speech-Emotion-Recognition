import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/features.csv")

print("=" * 70)
print("              SPEECH EMOTION MODEL TRAINING")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. SELECT FEATURES
# ============================================================

feature_columns = [f"mfcc_{i}" for i in range(1, 41)]

feature_columns += [
    "zcr",
    "rms",
    "spectral_centroid"
]

X = df[feature_columns]

y = df["emotion"]


# ============================================================
# 3. EXTRACT ACTOR / SPEAKER INFORMATION
# ============================================================

def get_actor(file_path):
    folder_name = os.path.basename(
        os.path.dirname(file_path)
    )
    return folder_name


groups = df["file"].apply(get_actor)


print("\nNumber of speakers:")
print(groups.nunique())

print("\nSpeakers:")
print(sorted(groups.unique()))


# ============================================================
# 4. ENCODE EMOTION LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nEmotion Classes:")
print(label_encoder.classes_)


# ============================================================
# 5. SPEAKER-INDEPENDENT TRAIN/TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y_encoded, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y_encoded[train_indices]
y_test = y_encoded[test_indices]


print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))


# ============================================================
# 6. SVM MODEL
# ============================================================

svm_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            random_state=42
        )
    )
])


print("\nTraining SVM...")
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)


# ============================================================
# 7. RANDOM FOREST MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)


print("Training Random Forest...")
random_forest.fit(X_train, y_train)

rf_predictions = random_forest.predict(X_test)


# ============================================================
# 8. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return accuracy, precision, recall, f1


# ============================================================
# 9. EVALUATE BOTH MODELS
# ============================================================

svm_scores = evaluate_model(
    "SUPPORT VECTOR MACHINE",
    y_test,
    svm_predictions
)

rf_scores = evaluate_model(
    "RANDOM FOREST",
    y_test,
    rf_predictions
)


# ============================================================
# 10. COMPARE MODELS
# ============================================================

print("\n" + "=" * 70)
print("                    MODEL COMPARISON")
print("=" * 70)

print(
    f"\nSVM Accuracy          : {svm_scores[0]:.4f}"
)

print(
    f"Random Forest Accuracy: {rf_scores[0]:.4f}"
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

if svm_scores[0] >= rf_scores[0]:

    best_model = svm_model
    best_predictions = svm_predictions
    best_model_name = "SVM"

else:

    best_model = random_forest
    best_predictions = rf_predictions
    best_model_name = "Random Forest"


print("\nBest Model:", best_model_name)


# ============================================================
# 12. CREATE MODELS FOLDER
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/emotion_model.pkl"
)

joblib.dump(
    label_encoder,
    "models/label_encoder.pkl"
)


print("\nBest model saved:")
print("models/emotion_model.pkl")

print("\nLabel encoder saved:")
print("models/label_encoder.pkl")


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)

display_labels = label_encoder.classes_

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "visualizations/confusion_matrix.png"
)

plt.show()


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("             MODEL TRAINING COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("✓ models/emotion_model.pkl")
print("✓ models/label_encoder.pkl")
print("✓ visualizations/confusion_matrix.png")

print("\nBest Model:", best_model_name)

print("=" * 70)