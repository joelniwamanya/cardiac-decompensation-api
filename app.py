import os
from fastapi import FastAPI, File, UploadFile
import torch
import torchaudio

# ---------------------------
# 1. Setup
# ---------------------------
app = FastAPI()

# Always use a relative path (works locally and on Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mcagnet_model.pth")

# Load model
model = torch.load(MODEL_PATH, map_location="cpu")
model.eval()

# ---------------------------
# 2. Routes
# ---------------------------
@app.get("/")
def home():
    return {"message": "Cardiac sound classification API is running!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav or .mp3).
    Returns: {"prediction": "Normal" or "Abnormal"}
    """
    waveform, sr = torchaudio.load(file.file)

    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize (must match training pipeline)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    # Predict
    with torch.no_grad():
        outputs = model(waveform.unsqueeze(0))
        pred_idx = torch.argmax(outputs, dim=1).item()

    label = "Abnormal" if pred_idx == 1 else "Normal"
    return {"prediction": label}
