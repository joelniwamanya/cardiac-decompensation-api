# ---- MC-AGNet (exact architecture that matches your checkpoint) ----
import torch
import torch.nn as nn
import torch.nn.functional as F

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
