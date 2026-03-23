import numpy as np
import torch
import torch.nn as nn
import math
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

# =============================================================================
# Paths & config
# =============================================================================
output_dir = 'model'
data_dir = 'Example_data/process_data'
save_dir = 'Command'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

seed = 42 

file_data_name   = "resultrealtime_log_2segments2_DP2_29_5_32.npz"

model_name_P     = 'resultrealtime_log_ros_20251108_205609P'
model_name_D     = 'resultrealtime_log_ros_20251108_214407D'
model_name_C     = 'crosstalkP2D_4_128_LSTM'
model_name_CD    = 'crosstalkD2P_4_128_LSTM'

file_sd_name_C   = "P2Dresultrealtime_log_ros_20251108_205609P_norm.npz"   # P->D crosstalk norm
file_sd_name_CD  = 'D2Presultrealtime_log_ros_20251108_214407D_norm.npz'   # D->P crosstalk norm
file_sd_name_P   = "resultrealtime_log_ros_20251108_205609P_norm.npz"
file_sd_name_D   = "resultrealtime_log_ros_20251108_214407D_norm.npz"

file_save_name_P  = model_name_P  + '.pth'
file_save_name_D  = model_name_D  + '.pth'
file_save_name_C  = model_name_C  + '.pth'   # P->D crosstalk model
file_save_name_CD = model_name_CD + '.pth'   # D->P crosstalk model

csv_out = model_name_P + 'shape_pressure_GT_1_7_1_5.csv'

sequence_length = 4
num_samples     = 80
dropout         = 0.1
H               = 4   # RSCC iterations

datapath   = os.path.join(data_dir, file_data_name)
normpath_P = os.path.join(data_dir, file_sd_name_P)
normpath_D = os.path.join(data_dir, file_sd_name_D)
normpath_C = os.path.join(data_dir, file_sd_name_C)
normpath_CD= os.path.join(data_dir, file_sd_name_CD)

modelpath_P  = os.path.join(output_dir, file_save_name_P)
modelpath_D  = os.path.join(output_dir, file_save_name_D)
modelpath_C  = os.path.join(output_dir, file_save_name_C)
modelpath_CD = os.path.join(output_dir, file_save_name_CD)

csvpath = os.path.join(save_dir, csv_out)

# =============================================================================
# Helpers
# =============================================================================
def compute_phi_features(xz):
    """ Helper to get [cos(phi), sin(phi)] for a 2D vector [x,z]. """
    x, z = xz
    phi = np.arctan2(z, x)
    return np.array([np.cos(phi), np.sin(phi)], dtype=np.float64)


def build_input_sequences_with_predictionP(
    trajP: np.ndarray,        # (T,2) proximal X,Z
    trajD: np.ndarray,        # (T,2) distal   X,Z (not used in features here)
    sequence_length: int,
    input_mean: np.ndarray,   # shape (8,)
    input_std:  np.ndarray,   # shape (8,)
    model:       torch.nn.Module,
    target_mean: np.ndarray,  # shape (3,)
    target_std:  np.ndarray,  # shape (3,)
    device:      torch.device
) -> np.ndarray:
    """
    Build proximal IK inputs and predict pressures along trajectory.

    Each per-timestep feature vector (length 8) is:
        [PX_prev, PZ_prev, cosP_prev, sinP_prev,
         dPX, dPZ, dcosP, dsinP]
    """
    T = trajP.shape[0]
    assert trajD.shape[0] == T, "Proximal and distal must be same length"

    # Pad at front with first row to allow first window
    padP = np.repeat(trajP[0:1], sequence_length - 1, axis=0)
    padD = np.repeat(trajD[0:1], sequence_length - 1, axis=0)  # kept for symmetry
    P = np.vstack([padP, trajP])   # (T + seq_len -1, 2)
    D = np.vstack([padD, trajD])   # not used in features, but kept

    feats = []
    for t in range(1, P.shape[0]):
        # current and previous for P
        curP   = P[t]
        prevP  = P[t-1]
        dP     = curP - prevP
        bP     = compute_phi_features(curP)
        bPprev = compute_phi_features(prevP)
        dbP    = bP - bPprev

        # stack into 8-vector
        vec = np.concatenate([
            prevP,     # 2
            bPprev,    # 2
            dP,        # 2
            dbP        # 2
        ])
        feats.append(vec)

    feats = np.stack(feats, axis=0)      # shape (T+seq_len-1, 8)
    num_wins = feats.shape[0] - sequence_length + 1

    X_seq = []
    for i in range(num_wins):
        window = feats[i : i + sequence_length]
        X_seq.append(window)

    X_seq = np.stack(X_seq, axis=0)     # (T, seq_len, 8) since num_wins == T
    # print("P X_seq shape:", X_seq.shape)

    # normalize
    X_seq = (X_seq - input_mean[None, None, :]) / input_std[None, None, :]

    # --- predict on each window ---
    preds = []
    model.eval()
    with torch.no_grad():
        for seq in X_seq:
            seq_tensor = torch.tensor(seq, dtype=torch.float32) \
                              .unsqueeze(0).to(device)   # (1, seq_len, 8)
            out = model(seq_tensor)                       # (1, 3)
            pt  = out.squeeze(0).cpu().numpy()            # (3,)
            pt  = pt * target_std + target_mean           # de-normalize
            preds.append(pt)

    predictions = np.stack(preds, axis=0)  # (T, 3)
    return predictions


def generate_circle_trajectory(radius, num_points=100, center=[0.0, 0.0], clockwise=True):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    if not clockwise:
        angles = angles[::-1]
    X = center[0] + radius * np.cos(angles)
    Z = center[1] + radius * np.sin(angles)
    return np.stack([X, Z], axis=1)


def generate_circle_trajectory2(radius, num_points=100, center=[0.0, 0.0], clockwise=True):
    angles = np.linspace(np.pi, 3 * np.pi, num_points, endpoint=False)
    if not clockwise:
        angles = angles[::-1]
    X = center[0] + radius * np.cos(angles)
    Z = center[1] + radius * np.sin(angles)
    return np.stack([X, Z], axis=1)


def build_input_sequences_with_predictionD(
    trajP: np.ndarray,        # (T,2) proximal X,Z (not used in features here)
    trajD: np.ndarray,        # (T,2) distal   X,Z
    sequence_length: int,
    input_mean: np.ndarray,   # shape (8,)
    input_std:  np.ndarray,   # shape (8,)
    model:       torch.nn.Module,
    target_mean: np.ndarray,  # shape (3,)
    target_std:  np.ndarray,  # shape (3,)
    device:      torch.device
) -> np.ndarray:
    """
    Build distal IK inputs and predict distal pressures.

    Each per-timestep feature vector (length 8) is:
        [DX_prev, DZ_prev, cosD_prev, sinD_prev,
         dDX, dDZ, dcosD, dsinD]
    """
    T = trajP.shape[0]
    assert trajD.shape[0] == T, "Proximal and distal must be same length"

    # pad at front with first row to allow first window
    padP = np.repeat(trajP[0:1], sequence_length - 1, axis=0)  # not used in features
    padD = np.repeat(trajD[0:1], sequence_length - 1, axis=0)
    P = np.vstack([padP, trajP])   # (T + seq_len -1, 2)
    D = np.vstack([padD, trajD])

    feats = []
    for t in range(1, D.shape[0]):
        curD   = D[t]
        prevD  = D[t-1]
        dD     = curD - prevD
        bD     = compute_phi_features(curD)
        bDprev = compute_phi_features(prevD)
        dbD    = bD - bDprev

        vec = np.concatenate([
            prevD, bDprev,
            dD,   dbD
        ])
        feats.append(vec)

    feats = np.stack(feats, axis=0)   # (T+seq_len-1, 8)
    num_wins = feats.shape[0] - sequence_length + 1

    X_seq = []
    for i in range(num_wins):
        window = feats[i : i + sequence_length]
        X_seq.append(window)

    X_seq = np.stack(X_seq, axis=0)   # (T, seq_len, 8)
    # print("D X_seq shape:", X_seq.shape)

    # normalize
    X_seq = (X_seq - input_mean[None, None, :]) / input_std[None, None, :]

    # --- predict on each window ---
    preds = []
    model.eval()
    with torch.no_grad():
        for seq in X_seq:
            seq_tensor = torch.tensor(seq, dtype=torch.float32) \
                              .unsqueeze(0).to(device)   # (1, seq_len, 8)
            out = model(seq_tensor)                       # (1, 3)
            pt  = out.squeeze(0).cpu().numpy()            # (3,)
            pt  = pt * target_std + target_mean           # de-normalize
            preds.append(pt)

    predictions = np.stack(preds, axis=0)  # (T, 3)
    return predictions


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=3, num_layers=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # output at last time step


def predict_crosstalk_traj(press_traj,  # shape (N, 3)
                           sequence_length,
                           model,
                           input_mean, input_std,
                           target_mean, target_std,
                           device):
    """
    Given a pressure trajectory (N x 3), return crosstalk offsets eta (N x 2).
    Uses same sequence_length as IK models.
    """
    # 1) normalize
    x_norm = (press_traj - input_mean) / input_std

    # 2) build sequences
    seqs = []
    for i in range(len(x_norm) - sequence_length + 1):
        seqs.append(x_norm[i:i + sequence_length])
    seqs = np.array(seqs)                     # (N - L + 1, L, 3)

    # 3) run LSTM
    preds = []
    model.eval()
    with torch.no_grad():
        for seq in seqs:
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)
            out = model(seq_tensor)          # (1, 2)
            preds.append(out.squeeze(0).cpu().numpy())
    preds = np.array(preds)                  # (N - L + 1, 2)

    # 4) denormalize target
    preds = preds * target_std + target_mean

    # 5) pad at the beginning so length matches N
    #    (reuse first estimate for the first L-1 samples)
    if sequence_length > 1:
        pad = np.repeat(preds[0:1], sequence_length - 1, axis=0)
        eta_full = np.vstack([pad, preds])
    else:
        eta_full = preds

    return eta_full   # (N, 2)

# =============================================================================
# Load models & norms
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Proximal IK model ---
norm_P = np.load(normpath_P)
input_mean_P = norm_P['input_mean']
input_std_P  = norm_P['input_std']
target_mean_P = norm_P['target_mean']
target_std_P  = norm_P['target_std']

model_P = LSTMModel(input_size=8, hidden_size=128, output_size=3, num_layers=3).to(device)
model_P.load_state_dict(torch.load(modelpath_P, map_location=device))
model_P.eval()

# --- Distal IK model ---
norm_D = np.load(normpath_D)
input_mean_D = norm_D['input_mean']
input_std_D  = norm_D['input_std']
target_mean_D = norm_D['target_mean']
target_std_D  = norm_D['target_std']

model_D = LSTMModel(input_size=8, hidden_size=128, output_size=3, num_layers=3).to(device)
model_D.load_state_dict(torch.load(modelpath_D, map_location=device))
model_D.eval()

# --- Crosstalk P -> D ---
norm_C = np.load(normpath_C)
input_mean_C = norm_C['input_mean']   # shape (3,)
input_std_C  = norm_C['input_std']
target_mean_C = norm_C['target_mean'] # shape (2,)
target_std_C  = norm_C['target_std']

model_C_P2D = LSTMModel(
    input_size=3,
    hidden_size=128,
    output_size=2,
    num_layers=2,
    dropout=dropout
).to(device)
model_C_P2D.load_state_dict(torch.load(modelpath_C, map_location=device))
model_C_P2D.eval()

# --- Crosstalk D -> P ---
norm_C_D2P = np.load(normpath_CD)
input_mean_C_D2P = norm_C_D2P['input_mean']
input_std_C_D2P  = norm_C_D2P['input_std']
target_mean_C_D2P = norm_C_D2P['target_mean']
target_std_C_D2P  = norm_C_D2P['target_std']

model_C_D2P = LSTMModel(
    input_size=3,
    hidden_size=128,
    output_size=2,
    num_layers=2,
    dropout=dropout
).to(device)
model_C_D2P.load_state_dict(torch.load(modelpath_CD, map_location=device))
model_C_D2P.eval()

# (You were loading data and picking random samples; Y_g is not used for this
# circle trajectory command generation, so I keep the loading but don't use it.)
data = np.load(datapath)
inputs_b = data['inputs']
targets_b = data['targets']
targets = targets_b[:, 0:6]

max_start = len(inputs_b) - sequence_length + 1
np.random.seed(seed)
rng_index = np.random.choice(max_start, size=num_samples, replace=False)
print("Random indices (not used in this script, but kept):", rng_index)

# =============================================================================
# Build base trajectories (desired Px,Pz and Dx,Dz)
# =============================================================================
singleP = generate_circle_trajectory(1.7, num_points=30, clockwise=True)
singleD = generate_circle_trajectory2(1.5, num_points=30, clockwise=True)

circle_trajP = np.vstack([singleP, singleP, singleP, singleP])   # (T,2)
circle_trajD = np.vstack([singleD, singleD, singleD, singleD])   # (T,2)
T = circle_trajP.shape[0]
assert circle_trajD.shape[0] == T

# Original targets (task space)
trajP0 = circle_trajP.copy()
trajD0 = circle_trajD.copy()

# Corrected targets (start from original)
trajP_corr = trajP0.copy()
trajD_corr = trajD0.copy()

# =============================================================================
# RSCC loop (H iterations)
# =============================================================================
for h in range(H):
    print(f"RSCC iteration {h+1}/{H}")

    # (1) Proximal IK: trajP_corr -> q1^h
    predictions_P = build_input_sequences_with_predictionP(
        trajP_corr,
        trajD_corr,          # not used in features, but provided
        sequence_length,
        input_mean_P,
        input_std_P,
        model_P,
        target_mean_P,
        target_std_P,
        device
    )   # (T,3)

    # Enforce "third-rank = 0" rule (smallest chamber = 0)
    min_idx = np.argmin(predictions_P, axis=1)  # (T,)
    rows    = np.arange(predictions_P.shape[0])
    predictions_P[rows, min_idx] = 0.0

    # (2) Crosstalk P -> D: q1^h -> eta_D^h
    eta_D = predict_crosstalk_traj(
        press_traj=predictions_P,
        sequence_length=sequence_length,
        model=model_C_P2D,
        input_mean=input_mean_C,
        input_std=input_std_C,
        target_mean=target_mean_C,
        target_std=target_std_C,
        device=device
    )   # (T,2)

    # (3) Distal IK: compensated distal targets -> q2^h
    # According to the paper: x2^(h) = x2^0 - eta2^(h)
    trajD_for_ik = trajD0 - eta_D

    predictions_D = build_input_sequences_with_predictionD(
        trajP_corr,       # proximal task (not used in features)
        trajD_for_ik,     # crosstalk-compensated distal targets
        sequence_length,
        input_mean_D,
        input_std_D,
        model_D,
        target_mean_D,
        target_std_D,
        device
    )   # (T,3)

    # (4) Crosstalk D -> P: q2^h -> eta_P^h
    eta_P = predict_crosstalk_traj(
        press_traj=predictions_D,
        sequence_length=sequence_length,
        model=model_C_D2P,
        input_mean=input_mean_C_D2P,
        input_std=input_std_C_D2P,
        target_mean=target_mean_C_D2P,
        target_std=target_std_C_D2P,
        device=device
    )   # (T,2)

    # (5) Update corrected task-space targets for next iteration:
    # x1^(h+1) = x1^0 - eta1^(h)
    # x2^(h+1) = x2^0 - eta2^(h)
    trajP_corr = trajP0 - eta_P
    trajD_corr = trajD0 - eta_D

# After RSCC loop, use final iteration outputs as command pressures
final_pressures_prox = predictions_P  # q1^H, shape (T,3)
final_pressures_dist = predictions_D  # q2^H, shape (T,3)

# =============================================================================
# Build CSV & plots
# =============================================================================
P1 = final_pressures_prox[:, 0]
P2 = final_pressures_prox[:, 1]
P3 = final_pressures_prox[:, 2]

P4 = final_pressures_dist[:, 0]
P5 = final_pressures_dist[:, 1]
P6 = final_pressures_dist[:, 2]

PX = trajP0[:, 0]
PZ = trajP0[:, 1]
PY = np.zeros(T)    # not used, filler

DX = trajD0[:, 0]
DY = trajD0[:, 1]
DZ = np.zeros(T)

df = pd.DataFrame({
    'P1': P1,
    'P2': P2,
    'P3': P3,
    'P4': P4,
    'P5': P5,
    'P6': P6,
    'PX': PX,
    'PY': PY,
    'PZ': PZ,
    'DX': DX,
    'DY': DY,
    'DZ': DZ,
})

df.to_csv(csvpath, index=False)
print(f"Saved RSCC commands to {csvpath}")

# Simple plot to inspect pressures (no GT here because trajectory is synthetic)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

for i in range(3):
    ax1.plot(df[f'P{i+1}'], '--', label=f'P{i+1}')
ax1.set_ylabel("Pressure [prox]")
ax1.set_title("Proximal segment pressures (RSCC)")
ax1.grid(True)
ax1.legend(loc='upper right', ncol=3)

for i in range(3, 6):
    ax2.plot(df[f'P{i+1}'], '--', label=f'P{i+1}')
ax2.set_xlabel("Sample index")
ax2.set_ylabel("Pressure [dist]")
ax2.set_title("Distal segment pressures (RSCC)")
ax2.grid(True)
ax2.legend(loc='upper right', ncol=3)

plt.tight_layout()
plt.show()
