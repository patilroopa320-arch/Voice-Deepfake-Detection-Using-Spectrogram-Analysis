# 🎤 Voice Deepfake Detection using Spectrogram Analysis

This project is a web-based application that analyzes uploaded audio files to determine whether the voice is spoken by a real human or generated using AI (deepfake) technology, using advanced deep learning methods.

## 🚀 Features
- Supports multiple audio formats: WAV, MP3, AAC, M4A
- Converts audio to a standard format automatically
- Splits long audio into smaller segments for accurate analysis
- Generates Mel-Spectrograms for feature extraction
- Uses a CNN model to classify speech as Human or AI-generated
- Provides confidence score, reliability level, and explanation
- Displays spectrogram image for visual verification

## 🛠️ Technologies Used
- Python (Flask)
- TensorFlow / Keras
- Librosa
- CNN Deep Learning Model
- HTML, CSS, JavaScript
- FFmpeg

## ⚙️ How It Works
1. User uploads an audio file
2. Audio is validated and converted to WAV format
3. Audio is split into segments
4. Spectrograms are generated
5. CNN model predicts each segment
6. Final result is aggregated and displayed to the user

## 📌 Applications
- Voice Deepfake Detection
- Digital Forensics
- Media Verification
- Cybersecurity
- Voice Authentication Systems

## ▶️ Run the Project
```bash
pip install -r requirements.txt
python app.py
