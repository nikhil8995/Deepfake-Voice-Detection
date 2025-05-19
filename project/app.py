import os
import torch
import torchaudio
import numpy as np
from flask import Flask, request, render_template
from model import CNNVoiceDetector, GenderCNN, get_mfcc_transform  # Make sure GenderCNN is imported

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Deepfake Detector Model
df_model = CNNVoiceDetector().to(device)
df_model.load_state_dict(torch.load("deepfake_voice_detector.pth", map_location=device))
df_model.eval()

# Load Gender Detector Model
gender_model = GenderCNN().to(device)
gender_model.load_state_dict(torch.load("gender_detector.pth", map_location=device))
gender_model.eval()

mfcc_transform = get_mfcc_transform()
max_frames = 100

def preprocess_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    mfcc = mfcc_transform(waveform)
    if mfcc.size(2) < max_frames:
        mfcc = torch.nn.functional.pad(mfcc, (0, max_frames - mfcc.size(2)))
    else:
        mfcc = mfcc[:, :, :max_frames]
    return mfcc

def predict_fake(file_path):
    mfcc = preprocess_audio(file_path)
    mfcc_tensor = mfcc.to(device).unsqueeze(0)
    with torch.no_grad():
        outputs = df_model(mfcc_tensor)
        _, pred = torch.max(outputs, 1)
    fake_result = "Real" if pred.item() == 0 else "Fake"
    return fake_result

def predict_gender(file_path):
    mfcc = preprocess_audio(file_path)
    mfcc_tensor = mfcc.to(device).unsqueeze(0)
    with torch.no_grad():
        outputs = gender_model(mfcc_tensor)
        _, pred = torch.max(outputs, 1)
    gender_result = "Female" if pred.item() == 0 else "Male"
    return gender_result

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files:
            result = "No file uploaded"
        else:
            file = request.files['file']
            if file.filename == '':
                result = "No file selected"
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                try:
                    fake_result = predict_fake(file_path)
                    gender_result = predict_gender(file_path)
                    result = f"Prediction: {fake_result}, Gender: {gender_result}"
                except Exception as e:
                    result = f"Error: {e}"
                os.remove(file_path)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)