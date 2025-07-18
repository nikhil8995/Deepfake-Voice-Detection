from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework import status
import torch
import librosa
import numpy as np
import os
from django.conf import settings
import torch.nn.functional as F

# Import the model definition
from deepfake_gender_cnn import SimpleCNN, infer_gender_from_filename, N_MELS, SAMPLE_RATE, DURATION

MODEL_PATH = os.path.join(settings.BASE_DIR, 'cnn_deepfake_gender.pth')

def load_model():
    model = SimpleCNN(N_MELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

class PredictView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        try:
            file_obj = request.FILES.get('audio')
            if not file_obj:
                return Response({'error': 'No audio file provided.'}, status=status.HTTP_400_BAD_REQUEST)
            # Save to temp file
            temp_path = os.path.join(settings.BASE_DIR, 'temp_upload.wav')
            with open(temp_path, 'wb+') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)
            # Preprocess audio
            y, sr = librosa.load(temp_path, sr=SAMPLE_RATE)
            length = SAMPLE_RATE * DURATION
            if len(y) < length:
                y = np.pad(y, (0, length - len(y)))
            else:
                y = y[:length]
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            mel_db = mel_db[np.newaxis, np.newaxis, ...]  # (1, 1, n_mels, time)
            X = torch.tensor(mel_db, dtype=torch.float32)
            # Load model and predict
            model = load_model()
            with torch.no_grad():
                out_deepfake, out_gender = model(X)
                # Apply softmax to get probabilities
                probs_deepfake = F.softmax(out_deepfake, dim=1)
                probs_gender = F.softmax(out_gender, dim=1)
                
                conf_deepfake, pred_deepfake = torch.max(probs_deepfake, 1)
                conf_gender, pred_gender = torch.max(probs_gender, 1)

            os.remove(temp_path)
            return Response({
                'deepfake': 'FAKE' if pred_deepfake.item() == 1 else 'REAL',
                'gender': 'FEMALE' if pred_gender.item() == 1 else 'MALE',
                'confidence_deepfake': f'{conf_deepfake.item()*100:.2f}%',
                'confidence_gender': f'{conf_gender.item()*100:.2f}%',
            })
        except Exception as e:
            import traceback
            return Response({'error': str(e), 'trace': traceback.format_exc()}, status=500)

def home(request):
    return render(request, 'index.html')
