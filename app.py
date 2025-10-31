# app.py — corrected and defensive version
import os
import logging
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------
# Config
# ----------------------
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "mcagnet_model.pth")
NUM_FEATURES = int(os.environ.get("NUM_FEATURES", 4))  # ensure this matches your preprocessing
NUM_CLASSES = int(os.environ.get("NUM_CLASSES", 2))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", 64))
NUM_CHANNELS = int(os.environ.get("NUM_CHANNELS", 5))

# ----------------------
# FastAPI instance
# ----------------------
app = FastAPI(title="MC-AGNet Cardiac Health API", version="1.0")

# ----------------------
# Model classes (fixed __init__ typos)
# ----------------------
class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adjacency):
        support = self.linear(x)
        output = torch.bmm(adjacency.unsqueeze(0).repeat(x.size(0), 1, 1), support)
        return output

class MCAGNet(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_channels=5):
        super(MCAGNet, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        
        # Graph for microphone array geometry
        self.adjacency = self._build_microphone_graph()
        
        # Channel-specific feature encoders
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(num_channels)
        ])
        
        # Multi-channel attention fusion (beamforming)
        self.channel_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.2, batch_first=True
        )
        
        # Graph convolution layers
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def _build_microphone_graph(self):
        adj = torch.tensor([
            [1, 1, 0, 0, 1],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 1]
        ], dtype=torch.float32)
        adj = adj + torch.eye(5)
        degree = torch.diag(torch.sum(adj, dim=1))
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree))
        normalized_adj = degree_inv_sqrt @ adj @ degree_inv_sqrt
        return normalized_adj
    
    def forward(self, x):
        batch_size, seq_len, num_channels, num_features = x.shape
        channel_features = []
        for c in range(self.num_channels):
            channel_data = x[:, :, c, :]
            channel_avg = torch.mean(channel_data, dim=1)
            encoded = self.channel_encoders[c](channel_avg)
            channel_features.append(encoded.unsqueeze(1))
        multi_channel = torch.cat(channel_features, dim=1)
        fused, attention_weights = self.channel_attention(multi_channel, multi_channel, multi_channel)
        gcn1_out = F.relu(self.gcn1(fused, self.adjacency))
        gcn2_out = F.relu(self.gcn2(gcn1_out, self.adjacency))
        max_pool = torch.max(gcn2_out, dim=1)[0]
        mean_pool = torch.mean(gcn2_out, dim=1)
        combined = torch.cat([max_pool, mean_pool], dim=1)
        out = self.classifier(combined)
        return out, attention_weights

# ----------------------
# Safe state-dict utilities
# ----------------------
def _strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def _extract_state_dict(loaded):
    # Many checkpoints are either:
    # - a raw state_dict
    # - a dict with keys like 'state_dict' or 'model_state_dict' mapping to the state dict
    if isinstance(loaded, dict):
        # check common keys
        for candidate in ("state_dict", "model_state_dict", "sd"):
            if candidate in loaded:
                return loaded[candidate]
        # otherwise assume loaded is already a state dict
        return loaded
    else:
        # unexpected format
        raise ValueError("Checkpoint loaded is not a dict or state_dict-like object.")

# ----------------------
# Checkpoint loader (defensive)
# ----------------------
def load_mcagnet_checkpoint(
    ckpt_path=CHECKPOINT_PATH,
    num_features=NUM_FEATURES,
    num_classes=NUM_CLASSES,
    hidden_dim=HIDDEN_DIM,
    num_channels=NUM_CHANNELS,
    map_location="cpu",
    strict=False,           # default to False so app starts; we still return mismatches
):
    device = torch.device(map_location)
    model = MCAGNet(num_features, num_classes, hidden_dim, num_channels)
    loaded = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(loaded)
    # strip 'module.' prefix if present
    if any(k.startswith("module.") for k in list(state_dict.keys())):
        state_dict = _strip_module_prefix(state_dict)
    # attempt to load with requested strictness and capture result
    load_result = model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model, load_result

# ----------------------
# Startup: attempt to load model but do NOT crash app on failure
# ----------------------
GLOBAL_MODEL = None
GLOBAL_LOAD_RESULT = None
if os.path.exists(CHECKPOINT_PATH):
    try:
        GLOBAL_MODEL, GLOBAL_LOAD_RESULT = load_mcagnet_checkpoint(
            ckpt_path=CHECKPOINT_PATH,
            num_features=NUM_FEATURES,
            num_classes=NUM_CLASSES,
            hidden_dim=HIDDEN_DIM,
            num_channels=NUM_CHANNELS,
            map_location="cpu",
            strict=False,  # let it load even if there are mismatches, so API can start
        )
        logger.info(f"Model loaded from {CHECKPOINT_PATH} (strict=False). Load result: {GLOBAL_LOAD_RESULT}")
    except Exception as e:
        logger.exception(f"Model load failed at startup: {e}")
        GLOBAL_MODEL = None
else:
    logger.warning(f"Checkpoint not found at {CHECKPOINT_PATH}. App will start without a model.")

# ----------------------
# API schemas and endpoints
# ----------------------
class FeatureSequence(BaseModel):
    data: list
    class Config:
        extra = "allow"

@app.get("/")
def read_root():
    return {"message": "MC-AGNet Cardiac API is running."}

@app.get("/load-model-test")
def load_model_test():
    """
    Attempt to load checkpoint and return missing/unexpected keys info.
    Use this endpoint to inspect your checkpoint on the deployed instance.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        return {"status": "missing_checkpoint", "path": CHECKPOINT_PATH}
    try:
        model, result = load_mcagnet_checkpoint(
            ckpt_path=CHECKPOINT_PATH,
            num_features=NUM_FEATURES,
            num_classes=NUM_CLASSES,
            hidden_dim=HIDDEN_DIM,
            num_channels=NUM_CHANNELS,
            map_location="cpu",
            strict=False,
        )
        # result is a NamedTuple on PyTorch with missing_keys & unexpected_keys when strict=False
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        return {"status": "loaded_with_issues" if (missing or unexpected) else "loaded_ok",
                "missing_keys_count": len(missing),
                "unexpected_keys_count": len(unexpected),
                "missing_keys_sample": missing[:40],
                "unexpected_keys_sample": unexpected[:40]}
    except Exception as e:
        logger.exception("load-model-test failed")
        return {"status": "error", "error": str(e)}

@app.post("/predict")
def predict_cardiac_condition(input_data: FeatureSequence):
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    try:
        input_tensor_np = np.array(input_data.data, dtype=np.float32)
        input_tensor = torch.tensor(input_tensor_np, dtype=torch.float32).unsqueeze(0)
        device = next(GLOBAL_MODEL.parameters()).device
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output, _ = GLOBAL_MODEL(input_tensor)
        probs = F.softmax(output, dim=1).squeeze().cpu().tolist()
        pred = int(torch.argmax(output, dim=1).item())
        labels = ["Normal", "Abnormal"]
        return {"prediction_label": labels[pred], "prediction_class": pred, "probabilities": {labels[i]: float(probs[i]) for i in range(len(probs))}}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
