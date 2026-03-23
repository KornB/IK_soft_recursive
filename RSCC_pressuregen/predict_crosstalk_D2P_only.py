import numpy as np
import torch
import torch.nn as nn
import os
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
model_dir = 'Trained_models'
data_dir = 'Example_data/process_data'
save_dir = 'Command'
os.makedirs(save_dir, exist_ok=True)

sequence_length = 4

model_name = 'crosstalkD2P_4_128_LSTM'
norm_name  = 'D2Presultrealtime_log_ros_20251108_214407D_norm.npz'

model_path = os.path.join(model_dir, model_name + '.pth')
norm_path  = os.path.join(data_dir, norm_name)

# =============================================================================
# MODEL
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 128, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# =============================================================================
# HELPERS
# =============================================================================
def build_sequences(data, seq_len):
    seqs = []
    for i in range(len(data) - seq_len + 1):
        seqs.append(data[i:i+seq_len])
    return np.array(seqs)

def predict_crosstalk(press_traj, model, mean, std, t_mean, t_std, device):
    x = (press_traj - mean) / std
    seqs = build_sequences(x, sequence_length)

    preds = []
    model.eval()
    with torch.no_grad():
        for seq in seqs:
            seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(seq).cpu().numpy()[0]
            preds.append(out)

    preds = np.array(preds)
    preds = preds * t_std + t_mean

    pad = np.repeat(preds[0:1], sequence_length - 1, axis=0)
    return np.vstack([pad, preds])

# =============================================================================
# MAIN
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

norm = np.load(norm_path)

model = LSTMModel().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Example input pressure
T = 100
press_dist = np.tile([15, 5, 0], (T,1))

eta_P = predict_crosstalk(
    press_dist,
    model,
    norm['input_mean'],
    norm['input_std'],
    norm['target_mean'],
    norm['target_std'],
    device
)

df = pd.DataFrame(eta_P, columns=['eta_P_x','eta_P_y'])
df.to_csv(os.path.join(save_dir, 'crosstalk_D2P_only.csv'), index=False)

print("Saved crosstalk_D2P_only.csv")
