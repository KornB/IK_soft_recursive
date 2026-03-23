import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import csv
import math
import os
output_dir = 'model'
data_dir = 'data_sep'
model_name = 'resultrealtime_log_ros_20251104_234130T_10_2_256_4_300'
os.makedirs(output_dir, exist_ok=True)
# === Config ===c:\Users\user\Desktop\two_segment\usemodel\test_seq_path_2segments_D.py
file_data_name = "resultrealtime_log_ros_20251104_234130.npz"
file_sd_name = "resultrealtime_log_ros_20251104_234130_norm.npz"
file_save_name = model_name+'.pth'
des_model_name = model_name+'.csv'
sequence_length = 11
# hiddle_state = 256
# num_layers = 3
batch_size = 32
num_epochs = 100
d_model = 256
nhead = 4
num_layers = 2
dropout = 0.15
# batch_size = 64
num_epochs = 100
MAX_SAMPLES = 700                     # set your cap
datapath = os.path.join(data_dir, file_data_name)
normpath = os.path.join(data_dir, file_sd_name)
modelpath = os.path.join(output_dir, file_save_name)
despath = os.path.join(output_dir, des_model_name)
# === Load & Normalize ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.load(datapath)
inputs = data['inputs']
targets = data['targets']
noise = np.random.normal(scale=0.1, size=inputs.shape)
inputs = inputs + noise

norm = np.load(normpath)
input_mean = norm['input_mean']
input_std = norm['input_std']
target_mean = norm['target_mean']
target_std = norm['target_std']

inputs = (inputs - input_mean) / input_std
targets = (targets - target_mean) / target_std

# === Create sequences ===
X_seq, Y_seq = [], []
for i in range(len(inputs) - sequence_length):
    X_seq.append(inputs[i:i + sequence_length])
    Y_seq.append(targets[i + sequence_length - 1])

X_seq = np.array(X_seq)
Y_seq = np.array(Y_seq)

# ---- limit total samples here ----
rng = np.random.default_rng(42)         # reproducible
n = min(MAX_SAMPLES, len(X_seq))
idx = rng.choice(len(X_seq), size=n, replace=False)
X_seq = X_seq[idx]
Y_seq = Y_seq[idx]

X_train, X_val, Y_train, Y_val = train_test_split(X_seq, Y_seq, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).float()), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float()), batch_size=batch_size, shuffle=False)
# === Model ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderOnlyTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.input_proj(src) * math.sqrt(d_model)
        src = self.pos_encoder(src)
        enc_out = self.encoder(src)
        return self.out_proj(enc_out[:, -1, :])  # Use last token
# === LSTM Model ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # hn: (num_layers, batch, hidden_size)
        return self.fc(hn[-1])     # use output from last LSTM layer

# === Training ===
input_size = inputs.shape[1]
output_size = targets.shape[1]
# model = LSTMModel(input_size, hidden_size=hiddle_state, output_size=output_size,num_layers=num_layers).to(device)
model = EncoderOnlyTransformer(input_size, output_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1.0e-4)
criterion = nn.MSELoss()

train_losses, val_losses = [], []
best_val = float('inf')

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    # print(f"Epoch {epoch+1}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), modelpath)
        print("Model saved.")
        print(f"Epoch {epoch+1}, Train: {train_loss:.4f}, Val: {val_loss:.4f}")

# === Save model info ===
with open(despath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["InputSize", input_size])
    writer.writerow(["OutputSize", output_size])
    writer.writerow(["HiddenSize", 256])
    writer.writerow(["NumLayers", 2])
    writer.writerow(["Dropout", 0.1])
    writer.writerow(["SequenceLength", sequence_length])
loss_csv_path = os.path.join(output_dir, "loss_curve"+model_name+".csv")
with open(loss_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "TrainLoss", "ValLoss"])
    for i, (tr, val) in enumerate(zip(train_losses, val_losses), 1):
        writer.writerow([i, tr, val])
