# 🗣️ Voice Deepfake & Gender Detection Web App
A Flask-based application that utilizes deep learning models to detect voice deepfakes and identify the speaker's gender from an uploaded audio file. Built with PyTorch and MFCC-based audio preprocessing, this tool provides a simple web interface to evaluate audio (.wav,.mp3) files.

🚀 Features

🎙️ Voice Deepfake Detection – Classifies audio as either Real or Fake using a trained CNN.

🚻 Gender Detection – Predicts the speaker's gender as Male or Female.

📂 Simple file upload UI with Flask.

🔊 Audio preprocessing using MFCC features.

🔍 Fully integrated with pre-trained PyTorch models.

🧠 Model Training

The models used in this app are trained in the Jupyter notebook:

df_detection.ipynb

This notebook contains the full pipeline:

Dataset loading and preprocessing

CNN architecture definitions

Model training and evaluation

📦 Dataset
This project uses the dataset from Kaggle:

🔗 DeepVoice AI Detection ver2 – by Emily Heart : https://www.kaggle.com/code/emilyheart/deepvoice-ai-detection-ver2/input

This dataset contains audio samples labeled as either real or AI-generated deepfakes, suitable for training binary classification models for deepfake detection. It also includes gender information for each sample, used in training the gender classification model.

📌 Make sure you have a Kaggle account and have accepted the dataset’s rules before downloading.

🗂 Project Structure

df_detection.ipynb                 # Notebook for training the models

project-root/

│

├── app.py                         # Flask app for running predictions

├── model.py                       # CNN architectures and preprocessing function

├── requirements.txt               # All Python dependencies

│

├── deepfake_voice_detector.pth    # Trained model for deepfake detection

├── gender_detector.pth            # Trained model for gender detection

│

├── static/

│   ├── style.css                  # Optional custom styles

│   ├── script.js                  # Optional JS for frontend behavior

│   └── favicon.ico                # Web app icon

│

└── templates/

    └── index.html                 # Main HTML template for UI

1. Set Up Environment

Install dependencies:

pip install -r requirements.txt

Make sure the following libraries are included:

flask

torch

torchaudio

numpy

2. Run the App

python app.py

The app will be accessible at: http://127.0.0.1:5000

🧪 Usage

Start the app.

Upload a .wav/.mp3 file (mono or stereo).

Receive the prediction:

Example: Prediction: Real, Gender: Female

📝 Notes

Models expect audio sampled at 16kHz.

Audio is transformed using MFCC and padded/truncated to 100 frames.

Ensure .pth model files are placed in the root directory.

🙌 Acknowledgements

Dataset: Kaggle - DeepVoice AI Detection v2

PyTorch, torchaudio, Flask

Open-source contributors
