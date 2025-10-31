# app.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
import torchaudio

# =====================================================
# 1. DEFINE MC-AGNet ARCHITECTURE (Simplified Rebuild)
# =====================================================
class MCAGNet(nn.Module):
    """
    MC-AGNet: Multi-Channel Audio Graph Network
    A simplified deployment version — must match your training structure
    """
    def __init__(self, num_features=4, num_classes=2, hidden_dim=64, num_nodes=5):
        super(MCAGNet, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Graph node encoders
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(num_nodes)
        ])

        # Attention between cardiac auscultation points
        self.node_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.2, batch_first=True
        )

        # Temporal processing
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # Expect input: (batch, seq_len, num_nodes, num_features)
        batch_size, seq_len, num_nodes, num_features = x.shape
        temporal_outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :, :]
            node_encodings = []
            for n in range(self.num_nodes):
                node_feat = x_t[:, n, :]
                encoded = self.feature_encoders[n](node_feat)
                node_encodings.append(encoded.unsqueeze(1))
            nodes = torch.cat(node_encodings, dim=1)
            attended, _ = self.node_attention(nodes, nodes, nodes)
            node_agg = torch.mean(attended, dim=1)
            temporal_outputs.append(node_agg)

        temporal_stack = torch.stack(temporal_outputs, dim=1)
        gru_out, _ = self.temporal_gru(temporal_stack)
        final_state = gru_out[:, -1, :]
        max_pool = torch.max(gru_out, dim=1)[0]
        combined = torch.cat([final_state, max_pool], dim=1)
        out = self.classifier(combined)
        return out


# =====================================================
# 2. DEPLOYMENT SETUP
# =====================================================
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mcagnet_model.pth")

# Initialize and load model weights
model = MCAGNet()
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()


# =====================================================
# 3. ROUTES
# =====================================================
@app.get("/")
def root():
    return {"message": "MC-AGNet Heart Sound Classification API is live!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Upload an audio file (.wav or .mp3)
    Returns: {"prediction": "Normal" or "Abnormal"}
    """
    waveform, sr = torchaudio.load(file.file)

    # Ensure mono and normalize
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    # Dummy reshape: (batch=1, seq_len=1, num_nodes=5, num_features=4)
    # Adjust according to your preprocessing
    x = waveform.unsqueeze(0).unsqueeze(0)[:, :, :5, :4]  # placeholder for structure

    with torch.no_grad():
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())

    label = "Abnormal" if pred == 1 else "Normal"
    return {"prediction": label}
