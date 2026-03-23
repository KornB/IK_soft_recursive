import numpy as np
import torch
import torch.nn as nn
import math
import os
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
model_dir = 'Trained_models'
data_dir = 'Example_data/process_data'
save_dir = 'Command'
os.makedirs(save_dir, exist_ok=True)

model_name = 'resultrealtime_log_ros_20251108_214407D'
norm_name  = 'resultrealtime_log_ros_20251108_214407D_norm.npz'

sequence_length = 4

model_path = os.path.join(model_dir, model_name + '.pth')
norm_path  = os.path.join(data_dir, norm_name)

# =============================================================================
# MODEL
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=8, hidden_size=128, output_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =============================================================================
# FEATURES
# =============================================================================
def compute_phi_features(xz):
    x, z = xz
    phi = np.arctan2(z, x)
    return np.array([np.cos(phi), np.sin(phi)])

def build_features(traj, seq_len):
    pad = np.repeat(traj[0:1], seq_len - 1, axis=0)
    traj = np.vstack([pad, traj])

    feats = []
    for t in range(1, len(traj)):
        cur = traj[t]
        prev = traj[t-1]

        d = cur - prev
        b = compute_phi_features(cur)
        bprev = compute_phi_features(prev)
        db = b - bprev

        vec = np.concatenate([prev, bprev, d, db])
        feats.append(vec)

    feats = np.stack(feats)

    X = []
    for i in range(len(feats) - seq_len + 1):
        X.append(feats[i:i+seq_len])

    return np.array(X)

# =============================================================================
# TRAJECTORY
# =============================================================================
def generate_circle(radius=1.2, N=100):
    t = np.linspace(0, 2*np.pi, N)
    return np.stack([radius*np.cos(t), radius*np.sin(t)], axis=1)

# =============================================================================
# MAIN
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm = np.load(norm_path)
input_mean = norm['input_mean']
input_std  = norm['input_std']
target_mean = norm['target_mean']
target_std  = norm['target_std']

model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

traj = generate_circle()

X = build_features(traj, sequence_length)
X = (X - input_mean[None,None,:]) / input_std[None,None,:]

preds = []
with torch.no_grad():
    for seq in X:
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
        out = model(seq).cpu().numpy()[0]
        out = out * target_std + target_mean
        preds.append(out)

preds = np.array(preds)

df = pd.DataFrame(preds, columns=['P4','P5','P6'])
df.to_csv(os.path.join(save_dir, 'distal_only.csv'), index=False)

print("Saved distal_only.csv")
