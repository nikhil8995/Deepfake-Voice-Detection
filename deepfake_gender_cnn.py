import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# --- CONFIG ---
AUDIO_DIR_REAL = 'AUDIO/REAL'
AUDIO_DIR_FAKE = 'AUDIO/FAKE'
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 64
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

# --- GENDER INFERENCE ---
FEMALE_NAMES = {'margot', 'taylor'}
def infer_gender_from_filename(filename):
    name = os.path.basename(filename).split('-')[0].lower()
    return 1 if name in FEMALE_NAMES else 0  # 1: female, 0: male

# --- DATASET ---
class AudioDataset(Dataset):
    def __init__(self, real_dir, fake_dir, sample_rate, duration, n_mels, augment=True):
        self.files = []
        self.labels = []
        self.genders = []
        for f in glob.glob(os.path.join(real_dir, '*.wav')):
            self.files.append(f)
            self.labels.append(0)  # 0: real
            self.genders.append(infer_gender_from_filename(f))
        for f in glob.glob(os.path.join(fake_dir, '*.wav')):
            self.files.append(f)
            self.labels.append(1)  # 1: fake
            self.genders.append(infer_gender_from_filename(f))
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.length = sample_rate * duration
        self.augment = augment
        # Print dataset stats
        n_real = sum(1 for l in self.labels if l == 0)
        n_fake = sum(1 for l in self.labels if l == 1)
        print(f"Loaded dataset: {n_real} real, {n_fake} fake samples.")

    def __len__(self):
        return len(self.files)

    def augment_audio(self, y):
        # Randomly apply augmentation
        if random.random() < 0.3:
            y = y + 0.005 * np.random.randn(len(y))  # Add noise
        if random.random() < 0.3:
            y = librosa.effects.pitch_shift(y, sr=self.sample_rate, n_steps=random.choice([-2, 2]))
        if random.random() < 0.3:
            gain = random.uniform(0.7, 1.3)
            y = y * gain  # Volume change
        return y

    def __getitem__(self, idx):
        file = self.files[idx]
        y, sr = librosa.load(file, sr=self.sample_rate)
        if self.augment:
            y = self.augment_audio(y)
        if len(y) < self.length:
            y = np.pad(y, (0, self.length - len(y)))
        else:
            y = y[:self.length]
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)  # normalize
        mel_db = mel_db[np.newaxis, ...]  # (1, n_mels, time)
        return (
            torch.tensor(mel_db, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            torch.tensor(self.genders[idx], dtype=torch.long),
        )

# --- MODEL ---
class SimpleCNN(nn.Module):
    def __init__(self, n_mels, n_classes=2, n_gender=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.fc_deepfake = nn.Linear(64 * (n_mels // 8) * (94 // 8), n_classes)
        self.fc_gender = nn.Linear(64 * (n_mels // 8) * (94 // 8), n_gender)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc_deepfake(x), self.fc_gender(x)

# --- TRAINING ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = AudioDataset(AUDIO_DIR_REAL, AUDIO_DIR_FAKE, SAMPLE_RATE, DURATION, N_MELS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleCNN(N_MELS).to(device)
    # Compute class weights
    n_real = sum(1 for l in dataset.labels if l == 0)
    n_fake = sum(1 for l in dataset.labels if l == 1)
    total = n_real + n_fake
    weights = torch.tensor([total/(2*n_real), total/(2*n_fake)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y_deepfake, y_gender in tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            X, y_deepfake, y_gender = X.to(device), y_deepfake.to(device), y_gender.to(device)
            optimizer.zero_grad()
            out_deepfake, out_gender = model(X)
            loss_deepfake = criterion(out_deepfake, y_deepfake)
            loss_gender = nn.CrossEntropyLoss()(out_gender, y_gender)
            loss = loss_deepfake + loss_gender
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}')
    torch.save(model.state_dict(), 'cnn_deepfake_gender.pth')
    print('Model saved as cnn_deepfake_gender.pth')

    # After training, print confusion matrix on the training set
    from sklearn.metrics import confusion_matrix
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for X, y_deepfake, _ in loader:
            X = X.to(device)
            out_deepfake, _ = model(X)
            preds = torch.argmax(out_deepfake, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_deepfake.numpy())
    cm = confusion_matrix(all_labels, all_preds)
    print('Confusion matrix (rows: true, cols: pred):\n', cm)

if __name__ == '__main__':
    train() 