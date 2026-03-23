# RSCC Pressure Generation – Code for RAL 2026 submission

This repository contains data preparation, training, and inference scripts for the RAL 2026 paper “ML Hysteresis & Crosstalk Compensation for a Soft Manipulator.” It trains LSTM/Transformer models for inverse kinematics and crosstalk compensation, and generates RSCC pressure commands for two soft arm segments.

## Repository Layout
- environment.yml — conda environment (Python 3.9, PyTorch 2.6.0+cu126, NumPy/Pandas/Sklearn/Matplotlib).
- Data_sep/ — dataset builders (data_prepare_shift_frame_*.py) converting raw logs into normalized .npz files for proximal/distal IK and crosstalk.
- Train_code/ — training scripts:
  - train_seq_model_LSTM_P2.py, train_seq_model_LSTM_D2.py, train_seq_model_LSTM_PD_11.py — proximal/distal IK LSTMs.
  - train_seq_model_encoder_crosstalkP2D_LSTM.py, train_seq_model_encoder_crosstalkD2P_LSTM.py — crosstalk LSTMs.
  - train_seq_model_Transformer_P2.py — Transformer IK variant.
- RSCC_pressuregen/test_model_with_GT_PD_shape_LSTM_11_cross_H.py — RSCC loop generating pressure commands for a circular trajectory, using trained IK + crosstalk models.
- Trained_models/ — pretrained checkpoints (*.pth) for proximal/distal IK and P↔D crosstalk.
- Example_data/raw_data/ — sample ROS logs (realtime_log_*.csv).
- Example_data/process_data/ — example processed/normalized data (*.npz, *.csv).
- RAL_2025_ML_Hysteresis_Crosstalk_SoftMani_Re1_submitted_main.pdf — submitted paper PDF.

## Getting Started
1) Create environment  
   conda env create -f environment.yml  
   conda activate torchenv

Recommended units:
- Position: **mm**
- Pressure: **kPa** or **bar** (choose one and use it consistently)
- Angle: **rad** or **deg** (choose one and use it consistently)

---

# Sensor, Data, and Model Input Requirements

This repository uses raw tracked-sensor logs and pressure commands to build training datasets for the following models:

1. **Proximal pressure mapping model**  
   Predicts proximal chamber pressures `[P1, P2, P3]`

2. **Distal pressure mapping model**  
   Predicts distal chamber pressures `[P4, P5, P6]`

3. **Proximal-to-distal response model (P2D / crosstalk-related model)**  
   Predicts distal 2D tip response `[X, Y]` from proximal pressures `[P1, P2, P3]`

4. **Distal-to-proximal response model (D2P / crosstalk-related model, optional)**  
   If used, this model should predict proximal 2D tip response `[X, Y]` from distal pressures `[P4, P5, P6]`

---

## 1. Raw data requirement

The current preprocessing scripts expect a raw `.csv` log with the following column names.

### Required columns

```text
time
p1, p2, p3, p4, p5, p6
ks

0A_pos_x, 0A_pos_y, 0A_pos_z
0A_orient_x, 0A_orient_y, 0A_orient_z, 0A_orient_w

0B_pos_x, 0B_pos_y, 0B_pos_z
0B_orient_x, 0B_orient_y, 0B_orient_z, 0B_orient_w

0C_pos_x, 0C_pos_y, 0C_pos_z
0C_orient_x, 0C_orient_y, 0C_orient_z, 0C_orient_w
```

### Notes

- `time` is the timestamp column
- `p1` to `p6` are chamber pressures
- `ks` is the logged stiffness-related index
- `0A`, `0B`, and `0C` are tracked sensor streams
- each sensor must provide both position and quaternion orientation
- keep the raw column names exactly as above so the current scripts can read them directly

### Example raw row

```text
time,p1,p2,p3,p4,p5,p6,ks,0A_pos_x,0A_pos_y,0A_pos_z,0A_orient_x,0A_orient_y,0A_orient_z,0A_orient_w,...
1762606570.41,0,0,0,0,0,0,0,15.3788,29.1995,-210.7228,...
```

---

## 2. Sensor placement requirement

To reproduce the current preprocessing pipeline, the raw log must contain:

- one reference sensor
- one proximal segment-end sensor
- one distal segment-end sensor

In the current scripts, these are read from the three raw tracked streams `0A`, `0B`, and `0C`, then internally transformed into local features.

For the downstream control pipeline, the important states are the two segment-end tip states:

- proximal segment end
- distal segment end

<img width="1445" height="696" alt="image" src="https://github.com/user-attachments/assets/6dcf13d4-b652-4377-a799-28568b45c422" />

---

## 3. Preprocessing requirement

The models are not trained directly from all raw rows.

The preprocessing scripts first:

1. load the raw log
2. detect rows around pressure changes
3. shift the selected rows by `-2` samples to account for valve delay
4. estimate zero-reference poses from zero-pressure regions
5. transform the raw sensor poses into local segment features
6. save processed datasets as `.npz`, normalization statistics as `.npz`, and inspection tables as `.csv`

---

## 4. Proximal pressure mapping model

### Purpose

Predict proximal chamber pressures `[P1, P2, P3]`.

### Raw data trigger

Rows are selected when any of the proximal pressures changes:

```text
p1, p2, or p3
```

### Zero-reference region

The zero-reference is estimated from rows where:

```text
p1 = p2 = p3 = 0
```

### Training input

The current preprocessing script builds an 8-dimensional input vector:

```text
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
```

### Feature meaning

- `PX`, `PY` = previous proximal local tip features
- `cosP`, `sinP` = proximal bending-plane representation
- `dPX`, `dPY`, `dcosP`, `dsinP` = change from previous step

### Training target

```text
[P1, P2, P3]
```

### Saved processed CSV format

```text
PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP, P1, P2, P3
```

### Input/output shape

```text
X.shape = [N, 8]
Y.shape = [N, 3]
```

### Deployment input

```text
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
```

### Deployment output

```text
[P1, P2, P3]
```

---

## 5. Distal pressure mapping model

### Purpose

Predict distal chamber pressures `[P4, P5, P6]`.

### Raw data trigger

Rows are selected when any of the distal pressures changes:

```text
p4, p5, or p6
```

### Zero-reference region

The zero-reference is estimated from rows where:

```text
p4 = p5 = p6 = 0
```

### Training input

The distal preprocessing script builds the same 8-dimensional input structure:

```text
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
```

### Feature meaning

- `PX`, `PY` = previous distal local tip features
- `cosP`, `sinP` = distal bending-plane representation
- `dPX`, `dPY`, `dcosP`, `dsinP` = change from previous step

### Training target

```text
[P4, P5, P6]
```

### Saved processed CSV format

```text
PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP, P4, P5, P6
```

### Input/output shape

```text
X.shape = [N, 8]
Y.shape = [N, 3]
```

### Deployment input

```text
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
```

### Deployment output

```text
[P4, P5, P6]
```

---

## 6. Proximal-to-distal response model (P2D / crosstalk-related model)

### Purpose

Model the distal tip response caused by proximal actuation.

### Current implementation

In the current uploaded preprocessing script, this model is built as:

```text
[P1, P2, P3] -> [X, Y]
```

This means the current `P2D` dataset is a pressure-to-distal-response mapping, rather than a full state-based crosstalk model.

### Raw data trigger

The current `P2D` preprocessing script uses proximal-pressure change events:

```text
p1, p2, or p3
```

### Training input

```text
[P1, P2, P3]
```

### Training target

```text
[X, Y]
```

where `[X, Y]` is the processed local distal 2D tip response.

### Saved processed CSV format

```text
P1, P2, P3, X, Y
```

### Input/output shape

```text
X.shape = [N, 3]
Y.shape = [N, 2]
```

### Deployment input

```text
[P1, P2, P3]
```

### Deployment output

```text
[X, Y]
```

---

## 7. Distal-to-proximal response model (D2P / crosstalk-related model, optional)

### Purpose

Model the proximal tip response caused by distal actuation.

### Note

A `D2P` preprocessing script is not included in the currently uploaded files.  
If a symmetric reverse-direction crosstalk model is needed, the recommended dataset format is:

```text
[P4, P5, P6] -> [X, Y]
```

where `[X, Y]` represents the processed local proximal 2D tip response.

### Recommended raw data trigger

```text
p4, p5, or p6
```

### Recommended training input

```text
[P4, P5, P6]
```

### Recommended training target

```text
[X, Y]
```

### Recommended saved processed CSV format

```text
P4, P5, P6, X, Y
```

### Recommended input/output shape

```text
X.shape = [N, 3]
Y.shape = [N, 2]
```

### Recommended deployment input

```text
[P4, P5, P6]
```

### Recommended deployment output

```text
[X, Y]
```

---

## 8. RSCC pipeline requirement

The RSCC pipeline uses the segment-end states together with the trained models.

### Minimum sensing requirement

- one reference sensor
- one proximal segment-end sensor
- one distal segment-end sensor

### Minimum model requirement

At minimum, the current logic supports:

1. proximal pressure mapping model

   ```text
   [PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P1, P2, P3]
   ```

2. distal pressure mapping model

   ```text
   [PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P4, P5, P6]
   ```

3. proximal-to-distal response model

   ```text
   [P1, P2, P3] -> [X, Y]
   ```

Optional:

4. distal-to-proximal response model

   ```text
   [P4, P5, P6] -> [X, Y]
   ```

### State representation

For downstream control, the robot state is represented by the two segment-end tip states:

```text
s_prox = proximal segment-end tip state
s_dist = distal segment-end tip state
```

These states are transformed into the required model input features before inference.

---
