# app.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import soundfile as sf
import io
import base64
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)

# MC-AGNet Architecture Classes
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adjacency):
        support = self.linear(x)
        output = torch.bmm(adjacency.unsqueeze(0).repeat(x.size(0), 1, 1), support)
        return output

class MCAGNet(nn.Module):
    def __init__(self, num_features=4, num_classes=2, hidden_dim=64, num_channels=5):
        super(MCAGNet, self).__init__()
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.adjacency = self._build_microphone_graph()
        
        self.channel_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ) for _ in range(num_channels)
        ])
        
        self.channel_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, dropout=0.2, batch_first=True
        )
        
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
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
        normalized_adj = torch.mm(torch.mm(degree_inv_sqrt, adj), degree_inv_sqrt)
        return normalized_adj
    
    def forward(self, x):
        batch_size, seq_len, num_channels, num_features = x.shape
        channel_features = []
        
        for channel in range(self.num_channels):
            channel_data = x[:, :, channel, :]
            channel_avg = torch.mean(channel_data, dim=1)
            encoded = self.channel_encoders[channel](channel_avg)
            channel_features.append(encoded.unsqueeze(1))
        
        multi_channel = torch.cat(channel_features, dim=1)
        fused, _ = self.channel_attention(multi_channel, multi_channel, multi_channel)
        
        gcn1_out = F.relu(self.gcn1(fused, self.adjacency))
        gcn2_out = F.relu(self.gcn2(gcn1_out, self.adjacency))
        
        max_pool = torch.max(gcn2_out, dim=1)[0]
        mean_pool = torch.mean(gcn2_out, dim=1)
        
        combined = torch.cat([max_pool, mean_pool], dim=1)
        output = self.classifier(combined)
        
        return output

# Initialize model globally
print("Loading MC-AGNet model...")
model = MCAGNet()

# Check if model file exists
if os.path.exists('mcagnet_model.pth'):
    model.load_state_dict(torch.load('mcagnet_model.pth', map_location=torch.device('cpu')))
    print("Model loaded successfully!")
else:
    print("Warning: Model file not found. Using untrained model.")

model.eval()

# Feature extraction function
def extract_audio_features(audio_data, sample_rate=4000):
    """Extract features from audio data"""
    
    # Resample if necessary
    if sample_rate != 4000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=4000)
    
    features = []
    
    # Time domain features
    features.append(np.mean(np.abs(audio_data)))  # Mean amplitude
    features.append(np.std(audio_data))  # Standard deviation
    features.append(np.max(np.abs(audio_data)))  # Max amplitude
    features.append(np.sqrt(np.mean(audio_data**2)))  # RMS energy
    
    # Ensure we have exactly 4 features
    features = features[:4] if len(features) > 4 else features + [0] * (4 - len(features))
    
    return np.array(features, dtype=np.float32)

# API Routes
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'MC-AGNet Cardiac Decompensation Detection API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Check API health',
            '/predict': 'Submit audio for analysis',
            '/stats': 'Get usage statistics'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': 'MC-AGNet',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get audio data from request
        data = request.json
        
        if 'audio' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        audio_base64 = data.get('audio')
        audio_bytes = base64.b64decode(audio_base64)
        
        # Load audio using soundfile
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Extract features
        features = extract_audio_features(audio_data, sample_rate)
        
        # Create input tensor for MC-AGNet
        # Shape: (batch_size=1, seq_len=10, channels=5, features=4)
        input_tensor = torch.zeros(1, 10, 5, 4)
        
        # Fill tensor with features (simulating multiple recordings)
        for i in range(10):  # Time steps
            for j in range(5):  # Channels
                # Add slight variation to simulate different channels
                variation = np.random.normal(0, 0.05, 4)
                input_tensor[0, i, j, :] = torch.tensor(features + variation)
        
        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        # Calculate risk metrics
        risk_score = probabilities[0, 1].item()  # Probability of abnormal
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = 'HIGH'
        elif risk_score > 0.4:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Generate recommendation
        recommendation = get_recommendation(risk_level, risk_score)
        
        # Prepare response
        response = {
            'success': True,
            'prediction': int(prediction),
            'prediction_label': 'Abnormal' if prediction == 1 else 'Normal',
            'confidence': float(confidence),
            'risk_level': risk_level,
            'risk_score': float(risk_score),
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
        
        # Log the prediction (you can save this to a database)
        log_prediction(response)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get API usage statistics"""
    # In production, these would come from a database
    stats = {
        'totalScans': 1247,
        'highRisk': 89,
        'avgConfidence': 87.3,
        'responseTime': 124,
        'todayScans': 43,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(stats)

def get_recommendation(risk_level, risk_score):
    """Generate medical recommendation based on risk"""
    recommendations = {
        'HIGH': {
            0.9: "URGENT: Very high risk of cardiac decompensation detected. Seek immediate emergency medical attention.",
            0.8: "URGENT: High risk of cardiac decompensation detected. Visit emergency department immediately.",
            0.7: "WARNING: Significant risk detected. Seek medical attention within 2-4 hours."
        },
        'MEDIUM': {
            0.6: "CAUTION: Moderate risk detected. Schedule appointment with cardiologist within 24 hours.",
            0.5: "CAUTION: Some abnormalities detected. Contact your healthcare provider within 48 hours.",
            0.4: "Monitor closely. Schedule check-up with your doctor this week."
        },
        'LOW': {
            0.3: "Low risk detected. Continue regular monitoring and maintain scheduled appointments.",
            0.2: "Minimal risk detected. Maintain healthy lifestyle and regular check-ups.",
            0.0: "Normal cardiac function detected. Continue routine health monitoring."
        }
    }
    
    # Find appropriate recommendation
    for level, thresholds in recommendations.items():
        if risk_level == level:
            for threshold, message in sorted(thresholds.items(), reverse=True):
                if risk_score >= threshold:
                    return message
    
    return "Unable to determine risk. Please consult healthcare provider."

def log_prediction(prediction_data):
    """Log predictions for monitoring and analytics"""
    # In production, save to database
    # For now, save to a JSON file
    try:
        log_file = 'predictions_log.json'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(prediction_data)
        
        # Keep only last 1000 predictions
        logs = logs[-1000:]
        
        with open(log_file, 'w') as f:
            json.dump(logs, f)
    except:
        pass  # Silently fail logging

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting MC-AGNet API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
