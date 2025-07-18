# Deepfake & Gender Detection Audio Analyzer

A modern, mobile-friendly web application for detecting deepfake audio and predicting speaker gender using deep learning. Built with Django, PyTorch, and a beautiful responsive frontend.

---

## Features
- **Upload or Drag & Drop Audio**: Analyze any .wav file for deepfake detection and gender prediction.
- **Waveform Visualization**: See and play the uploaded audio with an interactive waveform.
- **Example Audios**: Instantly test the app with real and fake sample files.
- **Confidence Scores**: Get model confidence for both deepfake and gender predictions.
- **Dark/Light Mode**: Toggle between beautiful themes with an animated sun/moon switch.
- **Mobile-First Design**: Fully responsive and works great on all devices.

---

## Quickstart

### 1. Clone the repository
```bash
git clone <https://github.com/nikhil8995/Deepfake-Voice-Detection>
cd Deepfake-Voice-Detection
```

### 2. Setup Conda Environment
```bash
conda create -n dfvenv python=3.11
conda activate dfvenv
```

### 3. Install Dependencies
```bash
conda install numpy scipy scikit-learn matplotlib tqdm
pip install torch librosa soundfile django djangorestframework pillow
```

### 4. Train the Model
```bash
python deepfake_gender_cnn.py
```

### 5. Run the Server
```bash
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Visit `http://localhost:8000/` or `http://<your-ip>:8000/` in your browser.

---

## Project Structure
```
├── AUDIO/                # Audio dataset (real/fake)
├── detection/            # Django app for API and views
├── dfweb/                # Django project settings
├── static/               # Static files (example audios)
├── templates/            # Frontend HTML
├── deepfake_gender_cnn.py# Model training script
├── manage.py             # Django entrypoint
├── requirements.txt      # (Optional) All dependencies
└── README.md
```

---

## Credits
- **Frontend**: Bootstrap 5, FontAwesome, WaveSurfer.js
- **Backend**: Django, PyTorch, Librosa
- **Author**: [Your Name]

---

## License
MIT License 