# ---- MC-AGNet (exact architecture that matches your checkpoint) ----
import torch
import torch.nn as nn
import torch.nn.functional as F
app = FastAPI()

# app.py (Add this section to the end of your file)

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration Constants ---
# NOTE: Update these paths and parameters based on your deployment environment
CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "mcagnet_model.pth")
NUM_FEATURES = 41 # MUST MATCH the features you extracted (e.g., from your previous code)
NUM_CLASSES = 2
HIDDEN_DIM = 64
NUM_CHANNELS = 5 # AV, PV, TV, MV, Phc

# --- 1. Define FastAPI Instance ---
app = FastAPI(
    title="MC-AGNet Cardiac Health API",
    version="1.0",
    description="API for cardiac audio analysis using a GNN model."
)

# --- 2. Load Model Once on Startup ---
try:
    # We map to 'cpu' as it's safer for general deployment unless using a GPU-enabled service
    GLOBAL_MODEL = load_mcagnet_checkpoint(
        ckpt_path=CHECKPOINT_PATH,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_channels=NUM_CHANNELS,
        map_location="cpu",
    )
    logger.info(f"✅ Model loaded successfully from {CHECKPOINT_PATH}")

except Exception as e:
    logger.error(f"FATAL ERROR: Could not load model checkpoint at {CHECKPOINT_PATH}. Details: {e}")
    # You might want to let the app start but disable the endpoint, or crash early
    # For now, we'll set it to None and handle the error in the endpoint.
    GLOBAL_MODEL = None
    raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")


# --- 3. Define Input Schema (Pydantic) ---
# This schema must match the shape of the features the model expects.
# Based on your previous data extraction: 
# (batch_size, sequence_length, num_channels, num_features)
class FeatureSequence(BaseModel):
    # Assuming the input is a list of features, representing the sequence, channels, and features.
    # The actual data will likely be passed as a list of lists of lists.
    # Example: List[List[List[float]]]
    data: list
    # Add any other required context here, like patient_id, etc.
    
    # Example of how to define nested lists that match the tensor shape
    # The structure must match your tensor: (seq_len, num_channels, num_features)
    # The input data should be validated carefully in a real deployment
    # For simplicity, we use a generic list here, but should be stricter.

    class Config:
        # Allows fields that are not defined in the schema (temporarily helpful)
        extra = "allow" 

# --- 4. Define Prediction Endpoint ---
@app.post("/predict")
async def predict_cardiac_condition(input_data: FeatureSequence):
    """
    Performs cardiac condition prediction using the loaded MC-AGNet model.
    Input data must be a nested list representing the feature sequence.
    """
    if GLOBAL_MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or initialization failed.")

    try:
        # Convert input list to PyTorch tensor
        # Expected shape: (seq_len, num_channels, num_features)
        input_tensor_np = np.array(input_data.data, dtype=np.float32)
        
        # Add batch dimension: (1, seq_len, num_channels, num_features)
        input_tensor = torch.tensor(input_tensor_np, dtype=torch.float32).unsqueeze(0)

        # Ensure the model and tensor are on the same device
        device = next(GLOBAL_MODEL.parameters()).device
        input_tensor = input_tensor.to(device)

        # Inference
        with torch.no_grad():
            output, _ = GLOBAL_MODEL(input_tensor)
        
        # Post-processing
        probabilities = F.softmax(output, dim=1).squeeze().tolist()
        predicted_class = torch.argmax(output, dim=1).item()
        
        # Map class index to label
        labels = ["Normal", "Abnormal"]
        predicted_label = labels[predicted_class]

        return {
            "prediction_label": predicted_label,
            "prediction_class": predicted_class,
            "probabilities": {labels[i]: prob for i, prob in enumerate(probabilities)}
        }
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        # The exception might be due to incorrect input shape or types.
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed due to processing error. Check input format. Error: {e}"
        )

@app.get("/")
def read_root():
    """Simple health check endpoint."""
    return {"message": "MC-AGNet Cardiac API is running."}

class GCNLayer(nn.Module):
    """Graph Convolutional Layer"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adjacency):
        # x: (batch, num_nodes, in_features)
        support = self.linear(x)  # (batch, num_nodes, out_features)
        # adjacency: (num_nodes, num_nodes)
        output = torch.bmm(adjacency.unsqueeze(0).repeat(x.size(0), 1, 1), support)
        return output

class MCAGNet(nn.Module):
    """
    MC-AGNet: Multi-Channel Audio Graph Network
    For beamforming and source localization in microphone arrays
    """
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
        """5-microphone circular configuration"""
        adj = torch.tensor([
            [1, 1, 0, 0, 1],  # 1↔2,5
            [1, 1, 1, 0, 0],  # 2↔1,3
            [0, 1, 1, 1, 0],  # 3↔2,4
            [0, 0, 1, 1, 1],  # 4↔3,5
            [1, 0, 0, 1, 1]   # 5↔1,4
        ], dtype=torch.float32)
        adj = adj + torch.eye(5)
        degree = torch.diag(torch.sum(adj, dim=1))
        degree_inv_sqrt = torch.inverse(torch.sqrt(degree))
        normalized_adj = degree_inv_sqrt @ adj @ degree_inv_sqrt
        return normalized_adj
    
    def forward(self, x):
        """
        x: (batch_size, sequence_length, num_channels, num_features)
        """
        batch_size, seq_len, num_channels, num_features = x.shape
        
        # 1) Channel encoders (average over time, per channel)
        channel_features = []
        for c in range(self.num_channels):
            channel_data = x[:, :, c, :]            # (batch, seq, features)
            channel_avg = torch.mean(channel_data, dim=1)  # (batch, features)
            encoded = self.channel_encoders[c](channel_avg)  # (batch, hidden)
            channel_features.append(encoded.unsqueeze(1))     # (batch, 1, hidden)
        
        multi_channel = torch.cat(channel_features, dim=1)     # (batch, C, hidden)
        
        # 2) Attention beamforming fusion
        fused, attention_weights = self.channel_attention(
            multi_channel, multi_channel, multi_channel
        )  # (batch, C, hidden)
        
        # 3) Graph convolution over channels-as-nodes
        gcn1_out = F.relu(self.gcn1(fused, self.adjacency))    # (batch, C, hidden)
        gcn2_out = F.relu(self.gcn2(gcn1_out, self.adjacency)) # (batch, C, hidden)
        
        # 4) Global pooling + classifier
        max_pool = torch.max(gcn2_out, dim=1)[0]   # (batch, hidden)
        mean_pool = torch.mean(gcn2_out, dim=1)    # (batch, hidden)
        combined = torch.cat([max_pool, mean_pool], dim=1)  # (batch, 2*hidden)
        out = self.classifier(combined)            # (batch, num_classes)
        return out, attention_weights


# ---- Safe loader that matches your checkpoint ----
def _strip_module_prefix(state_dict):
    # If saved with DataParallel, keys are like 'module.xxx'
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
    return new_sd

def load_mcagnet_checkpoint(
    ckpt_path="mcagnet_model.pth",
    num_features=4,
    num_classes=2,
    hidden_dim=64,
    num_channels=5,
    map_location="cpu",
):
    device = torch.device(map_location)
    model = MCAGNet(num_features, num_classes, hidden_dim, num_channels)
    sd = torch.load(ckpt_path, map_location=device)
    # strip module. prefixes if present (harmless if not)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = _strip_module_prefix(sd)
    # strict=True so we catch any mismatch early
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model
