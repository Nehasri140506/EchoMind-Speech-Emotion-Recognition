# EchoMind – Speech Emotion Recognition

## 1. Introduction

EchoMind is a Speech Emotion Recognition web application that analyzes an uploaded audio file and predicts the emotion expressed in the speech using a machine learning model. It provides the predicted emotion along with confidence and audio-related information through a simple web interface.

## 2. Features

- Upload and analyze audio files.
- Predict the emotion expressed in speech.
- Display the predicted emotion and confidence score.
- Display emotion probability values.
- Show audio information such as duration, sample rate, and sample count.
- Simple and user-friendly web interface.

## 3. Technologies Used

- **Programming Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** Trained emotion classification model
- **Audio Processing:** Librosa and NumPy (as used by the project)
- **Model Storage:** Pickle

## 4. Project Structure

```text
Speech_Emotion_Recognition/
├── models/
│   ├── emotion_model.pkl
│   └── label_encoder.pkl
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/
│   └── index.html
├── .gitignore
├── README.md
└── web_backend.py
```

## 5. Installation and Setup

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Speech_Emotion_Recognition
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### Step 2: Create a virtual environment

```bash
python -m venv venv
```

### Step 3: Activate the virtual environment on Windows

```bash
venv\Scripts\activate
```

### Step 4: Install the required packages

Install the packages listed in the project's `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
python web_backend.py
```

Open the following address in your browser:

http://127.0.0.1:5000/

## 6. How to Use

1. Start the Flask application.
2. Open the website in your browser.
3. Select a supported audio file.
4. Click the analyze button.
5. View the predicted emotion, confidence score, and available audio details.

## 7. Purpose

The purpose of EchoMind is to demonstrate how machine learning and audio feature extraction can be used to recognize emotions from speech through a web-based application.

## 8. Note

This application is intended for educational and demonstration purposes. Prediction results depend on the trained model, audio quality, and input speech.