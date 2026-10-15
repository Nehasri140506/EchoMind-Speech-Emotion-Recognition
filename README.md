# EchoMind – Speech Emotion Recognition

## 1. Introduction

EchoMind is a Speech Emotion Recognition web application that analyzes uploaded speech audio and predicts the emotion expressed using a trained machine learning model. It provides a simple web interface for uploading audio files and viewing prediction results.

This project demonstrates the use of machine learning, audio processing, and web development to recognize emotions from speech.

## 2. Objectives

- Recognize emotions expressed in speech audio.
- Extract relevant audio features for emotion classification.
- Predict emotions using a trained machine learning model.
- Display prediction results through a user-friendly web interface.

## 3. Features

- Upload audio files from your computer.
- Analyze speech audio using the trained model.
- Display the predicted emotion and confidence information.
- Show emotion probability values where supported by the application.
- Display audio information such as duration, sample rate, and sample count where supported.
- Simple and user-friendly web interface.

## 4. Technologies Used

- **Programming Language:** Python
- **Backend Framework:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Scikit-learn and a trained emotion classification model
- **Audio Processing:** Librosa and NumPy
- **Model Storage:** Pickle

## 5. Project Structure

```text
Speech_Emotion_Recognition/
├── app/
├── data/
├── dataset/
│   └── RAVDESS/
├── models/
│   ├── emotion_model.pkl
│   └── label_encoder.pkl
├── src/
│   ├── audio_analysis.py
│   ├── eda.py
│   ├── predict.py
│   ├── prepare_dataset.py
│   └── train_model.py
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── test_audio/
├── visualizations/
├── .gitignore
├── requirements.txt
├── web_backend.py
└── README.md
```

## 6. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/Nehasri140506/EchoMind-Speech-Emotion-Recognition.git
```

### Step 2: Open the project directory

```bash
cd EchoMind-Speech-Emotion-Recognition
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

### Step 4: Activate the virtual environment

On Windows Command Prompt:

```bat
venv\Scripts\activate.bat
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

## 7. Running the Application

Start the Flask application:

```bash
python web_backend.py
```

Open your browser and visit:

http://127.0.0.1:5000/

## 8. How It Works

1. The user uploads a speech audio file.
2. The backend receives and processes the audio.
3. Relevant audio features are extracted according to the implemented pipeline.
4. The trained machine learning model predicts the emotion.
5. The application displays the prediction and the available result information.

## 9. Dataset

The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset for speech emotion recognition.

Dataset access and redistribution are subject to the dataset's applicable terms and license.

## 10. Applications

- Speech emotion recognition demonstrations.
- Human-computer interaction research.
- Machine learning and audio-processing education.
- Experimental emotion-aware interfaces.

## 11. Limitations

Prediction results may vary depending on audio quality, background noise, the input speech, and the trained model's performance. Predicted emotions should be treated as estimates rather than definitive assessments of a person's emotional state.

## 12. Conclusion

EchoMind demonstrates how machine learning, audio feature extraction, and web development can be combined to create a speech emotion recognition application. It provides a practical project for exploring audio processing and emotion classification.

**Project Domain:** Machine Learning and Web Development