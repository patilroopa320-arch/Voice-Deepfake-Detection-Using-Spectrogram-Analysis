from flask import Flask, request, render_template, url_for, redirect
import os, uuid
import numpy as np
import librosa, librosa.display

# ==============================
# NON-GUI BACKEND
# ==============================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

from pydub import AudioSegment
from pydub.utils import which

# Set FFmpeg path
AudioSegment.converter = which("ffmpeg") or "/usr/bin/ffmpeg"

# ==============================
# CPU ONLY
# ==============================
tf.config.set_visible_devices([], "GPU")

# ==============================
# APP CONFIG
# ==============================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

UPLOAD_FOLDER = "uploads"
SPEC_FOLDER = "static/spectrograms"
MODEL_PATH = "model/deepfake_model.h5"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SPEC_FOLDER, exist_ok=True)

IMG_SIZE = 224
MIN_AUDIO_DURATION = 2.0
SEGMENT_SEC = 5

# 🔥 CALIBRATED THRESHOLDS (REALISTIC)
LOW_CONF = 0.40     # human
HIGH_CONF = 0.70    # AI

# ==============================
# LOAD MODEL
# ==============================
try:
    model = load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ==============================
# AUDIO → WAV
# ==============================
def convert_to_wav(path):
    wav_path = os.path.splitext(path)[0] + ".wav"
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        print(f"Conversion error: {e}")
        # Try loading directly with librosa as fallback
        try:
            y, sr = librosa.load(path, sr=16000)
            librosa.output.write_wav(wav_path, y, sr)
            return wav_path
        except:
            raise ValueError("Unable to decode audio. Please ensure the file is a valid audio format.")

# ==============================
# SPLIT AUDIO INTO SEGMENTS
# ==============================
def split_audio(y, sr):
    samples_per_seg = SEGMENT_SEC * sr
    segments = []

    for i in range(0, len(y), samples_per_seg):
        chunk = y[i:i + samples_per_seg]
        if len(chunk) > sr * 2:   # at least 2 sec
            segments.append(chunk)

    return segments

# ==============================
# GENERATE SPECTROGRAM
# ==============================
def generate_spec(y, sr, path):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db, sr=sr, cmap="magma")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()

# ==============================
# HOME PAGE (Landing Page)
# ==============================
@app.route("/")
def index():
    return render_template("index.html")

# ==============================
# DETECTION PAGE
# ==============================
@app.route("/detect")
def detect_page():
    return render_template("index.html", scroll_to="detect")

# ==============================
# PREDICTION ENDPOINT
# ==============================
@app.route("/predict", methods=["POST"])
def predict():
    # Check if model is loaded
    if model is None:
        return render_template("index.html", 
                             scroll_to="detect",
                             error="Model not loaded. Please check server configuration.")
    
    try:
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", 
                                 scroll_to="detect",
                                 error="Please select an audio file to analyze.")

        # Validate file extension
        allowed_extensions = {'.wav', '.mp3', '.m4a', '.aac', '.ogg'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return render_template("index.html",
                                 scroll_to="detect",
                                 error=f"Unsupported file type. Please upload: {', '.join(allowed_extensions)}")

        # Generate unique ID for this analysis
        uid = str(uuid.uuid4())
        raw_path = os.path.join(
            UPLOAD_FOLDER, uid + "_" + secure_filename(file.filename)
        )
        file.save(raw_path)

        # Convert to WAV if needed
        try:
            if file_ext != '.wav':
                wav_path = convert_to_wav(raw_path)
                os.remove(raw_path)  # Remove original to save space
                raw_path = wav_path
        except Exception as e:
            os.remove(raw_path)
            return render_template("index.html",
                                 scroll_to="detect",
                                 error=f"Failed to process audio file: {str(e)}")

        # Load and validate audio
        try:
            y, sr = librosa.load(raw_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            os.remove(raw_path)
            return render_template("index.html",
                                 scroll_to="detect",
                                 error=f"Failed to load audio: {str(e)}")

        if duration < MIN_AUDIO_DURATION:
            os.remove(raw_path)
            return render_template("index.html",
                                 scroll_to="detect",
                                 error=f"Audio too short. Minimum {MIN_AUDIO_DURATION} seconds required.")

        # Split into segments
        segments = split_audio(y, sr)
        if not segments:
            os.remove(raw_path)
            return render_template("index.html",
                                 scroll_to="detect",
                                 error="Insufficient voice content detected in audio.")

        # Process each segment
        votes = []
        preview_spec = os.path.join(SPEC_FOLDER, uid + ".png")

        for i, seg in enumerate(segments):
            temp_spec = os.path.join(SPEC_FOLDER, f"{uid}_{i}.png")
            generate_spec(seg, sr, temp_spec)

            # Load and preprocess spectrogram for model
            img = load_img(temp_spec, target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            # Get prediction
            pred = model.predict(img, verbose=0)[0][0]
            votes.append(pred)

            # Keep first spectrogram as preview, delete others
            if i == 0:
                os.replace(temp_spec, preview_spec)
            else:
                if os.path.exists(temp_spec):
                    os.remove(temp_spec)

        # Clean up audio file
        if os.path.exists(raw_path):
            os.remove(raw_path)

        # Calculate average prediction
        avg_pred = float(np.mean(votes))
        
        # ==============================
        # FINAL VERDICT (BALANCED)
        # ==============================
        if avg_pred >= HIGH_CONF:
            verdict = "AI-Generated Speech Detected ❌"
            explanation = (
                "Consistent synthetic spectral patterns detected "
                "across multiple audio segments. Characteristics typical "
                "of AI-generated speech were identified."
            )
            reliability = "High"
            confidence = round(avg_pred * 100, 2)

        elif avg_pred <= LOW_CONF:
            verdict = "Human Speech Detected ✅"
            explanation = (
                "Natural voice variations preserved with organic spectral patterns. "
                "Analysis aligns with human speech characteristics including "
                "natural harmonics and formant transitions."
            )
            reliability = "High"
            confidence = round((1 - avg_pred) * 100, 2)

        else:
            verdict = "Inconclusive Result ⚠️"
            explanation = (
                "Mixed indicators detected. This could be due to: "
                "1) Heavy audio compression or noise "
                "2) Voice conversion or editing "
                "3) Borderline AI generation quality. "
                "Manual verification recommended."
            )
            reliability = "Medium"
            confidence = round(max(avg_pred, 1 - avg_pred) * 100, 2)

        # Calculate sample rate info
        sample_rate = f"{sr:,} Hz"
        
        # Prepare result data
        result_data = {
            "prediction": verdict,
            "confidence": confidence,
            "reliability": reliability,
            "explanation": explanation,
            "duration": round(duration, 2),
            "sample_rate": sample_rate,
            "spectrogram_url": url_for("static", filename=f"spectrograms/{uid}.png"),
            "segments_analyzed": len(segments)
        }

        return render_template(
            "index.html",
            scroll_to="results",
            **result_data
        )

    except Exception as e:
        # Clean up any temporary files
        import traceback
        error_details = traceback.format_exc()
        print(f"Prediction error: {error_details}")
        
        return render_template("index.html",
                             scroll_to="detect",
                             error=f"Analysis failed: {str(e)}")

# ==============================
# ERROR HANDLERS
# ==============================
@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html',
                         scroll_to="detect",
                         error="Internal server error. Please try again."), 500

# ==============================
# HEALTH CHECK
# ==============================
@app.route("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# ==============================
# RUN APPLICATION
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)