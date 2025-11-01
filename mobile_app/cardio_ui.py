# cardio_ui.py
import streamlit as st
import requests
import numpy as np
import sounddevice as sd
import wavio
import tempfile
import json
import time

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
API_URL = "https://cardiac-decompensation-api.onrender.com/predict"
SAMPLE_RATE = 22050  # sampling rate for recording (Hz)
CHANNELS = 1         # mono recording
DURATION = 5         # default seconds to record

# ----------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------
st.set_page_config(page_title="CardioSense", page_icon="❤️", layout="centered")

st.title("🩺 CardioSense – Heart Sound Classifier")
st.markdown("Record your heart sound and let the AI model predict if it's **Normal** or **Abnormal.**")

# ----------------------------------------------------
# RECORD SECTION
# ----------------------------------------------------
st.subheader("🎙 Record Your Heart Sound")

duration = st.slider("Recording duration (seconds):", 3, 10, DURATION)
if st.button("Start Recording"):
    st.info("Recording... Please stay quiet and still.")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
    sd.wait()
    st.success("Recording finished!")
    
    # Save to temporary WAV file
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavio.write(tmp_wav.name, recording, SAMPLE_RATE, sampwidth=2)
    
    # Optional: playback preview
    st.audio(tmp_wav.name, format="audio/wav")

    # Store the recorded audio file path in Streamlit session
    st.session_state["audio_file"] = tmp_wav.name

# ----------------------------------------------------
# PREDICTION SECTION
# ----------------------------------------------------
if "audio_file" in st.session_state:
    st.subheader("⚙️ Analyze Your Recording")
    if st.button("Send to Model"):
        st.info("Extracting features and contacting the model...")

        # Simulated feature extraction (placeholder)
        # Replace this with your MFCC or spectrogram extraction later
        # (seq_len, num_channels, num_features)
        fake_features = np.random.rand(10, 5, 4).tolist()

        try:
            # Send POST request to Render API
            response = requests.post(API_URL, json={"data": fake_features}, timeout=30)
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Prediction complete!")
                
                # Display results
                label = result.get("prediction_label", "Unknown")
                probs = result.get("probabilities", {})
                st.markdown(f"### 🧠 Model Prediction: **{label}**")
                st.json(probs)
                
                # Probability visualization
                st.progress(list(probs.values())[0] if label == "Normal" else list(probs.values())[1])
            else:
                st.error(f"Server returned status code {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Error communicating with model: {e}")
else:
    st.warning("Please record your heart sound before sending it to the model.")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.caption("Powered by MC-AGNet • Deployed via FastAPI on Render • Built with Streamlit ❤️")
