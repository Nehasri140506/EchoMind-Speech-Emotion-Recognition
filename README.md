# EchoMind – Speech Emotion Recognition

## Project Overview

EchoMind is a web application that uses machine learning to predict emotions from uploaded speech audio files. It provides a simple interface for uploading audio and viewing the predicted emotion.

## Objectives

- Recognize emotions from speech audio.
- Extract relevant audio features.
- Predict emotions using a trained machine learning model.
- Display prediction results through a web interface.

## Features

- Upload audio files from your computer.
- Analyze speech audio.
- Predict emotions using a trained model.
- Display the predicted emotion and confidence information.
- View audio file details.

## Technologies Used

- Python
- Flask
- HTML
- CSS
- JavaScript
- Librosa
- NumPy
- Scikit-learn

## Project Structure

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
├── web_backend.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/EchoMind-Speech-Emotion-Recognition.git
   ```

2. Open the project directory:

   ```bash
   cd EchoMind-Speech-Emotion-Recognition
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate it on Windows Command Prompt:

   ```bat
   venv\Scripts\activate.bat
   ```

5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

Start the Flask server:

```bash
python web_backend.py
```

Open your browser and visit:

http://127.0.0.1:5000/

## How It Works

1. The user uploads a speech audio file.
2. The backend processes the audio.
3. Relevant audio features are extracted.
4. The trained model predicts the emotion.
5. The result is displayed on the website.

## Conclusion

EchoMind demonstrates the use of machine learning and web development to build a speech emotion recognition application.

**Project Domain:** Machine Learning and Web Development
