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

# Data Requirements

This section describes the required sensor setup and data formats for training and deployment of the models used in this repository. The framework includes:

1. **Pressure Mapping Model**  
   Maps segment tip states or target states to chamber pressure commands.

2. **Crosstalk Model**  
   Predicts inter-segment coupling effects caused by the motion or pressurization of another segment.

3. **RSCC Pipeline**  
   Recursive Segment-wise Crosstalk Compensation pipeline for multi-segment control using segment-end tip positions.

---

## 1. Sensor Setup

### 1.1 Required sensors
The system requires tip position sensing for each segment endpoint. Depending on the experimental setup, this can be obtained using:
- EM tracking sensors
- optical tracking
- vision-based tracking
- any equivalent pose sensing system with sufficient accuracy

### 1.2 Sensor placement
For a two-segment manipulator, the minimum required sensing locations are:

- **Sensor A** at the end of **Segment 1**
- **Sensor B** at the end of **Segment 2** (distal tip)

These two sensor measurements define:
- the endpoint of the proximal segment
- the final endpoint of the distal segment

<img width="1445" height="696" alt="image" src="https://github.com/user-attachments/assets/6dcf13d4-b652-4377-a799-28568b45c422" />

This is the minimum sensing requirement for the **RSCC pipeline**.

### 1.3 Coordinate frame
All recorded positions must be expressed in a **consistent global frame** or a clearly defined robot base frame.  
The same frame must be used:
- during dataset collection
- during model training
- during deployment

### 1.4 Sampling considerations
- Sensor and pressure command data should be time-aligned
- All channels should be sampled at a consistent rate, or resampled offline before training
- Missing samples should be removed or interpolated consistently
- Units must remain consistent across all files

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
Notes
time is the timestamp column
p1 to p6 are chamber pressures
ks is the logged stiffness-related index
0A, 0B, and 0C are tracked sensor streams
each sensor must provide both position and quaternion orientation
keep the raw column names exactly as above so the current scripts can read them directly
Example raw row
time,p1,p2,p3,p4,p5,p6,ks,0A_pos_x,0A_pos_y,0A_pos_z,0A_orient_x,0A_orient_y,0A_orient_z,0A_orient_w,...
1762606570.41,0,0,0,0,0,0,0,15.3788,29.1995,-210.7228,...
2. Sensor placement requirement

To reproduce the current preprocessing pipeline, the raw log must contain:

one reference sensor
one proximal segment-end sensor
one distal segment-end sensor

In the current scripts, these are read from the three raw tracked streams 0A, 0B, and 0C, then internally transformed into local features.

For the downstream control pipeline, the important states are the two segment-end tip states:

proximal segment end
distal segment end
3. Preprocessing requirement

The models are not trained directly from all raw rows.

The preprocessing scripts first:

load the raw log
detect rows around pressure changes
shift the selected rows by -2 samples to account for valve delay
estimate zero-reference poses from zero-pressure regions
transform the raw sensor poses into local segment features
save processed datasets as .npz, normalization stats as .npz, and inspection tables as .csv
4. Proximal pressure mapping model
Purpose

Predict proximal chamber pressures [P1, P2, P3].

Raw data trigger

Rows are selected when any of the proximal pressures changes:

p1, p2, or p3
Training input

The current preprocessing script builds an 8-dimensional input vector:

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Feature meaning
PX, PY = previous proximal local tip features
cosP, sinP = proximal bending-plane representation
dPX, dPY, dcosP, dsinP = change from previous step
Training target
[P1, P2, P3]
Saved processed CSV format
PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP, P1, P2, P3
Input/output shape
X.shape = [N, 8]
Y.shape = [N, 3]
Deployment input
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Deployment output
[P1, P2, P3]
5. Distal pressure mapping model
Purpose

Predict distal chamber pressures [P4, P5, P6].

Raw data trigger

Rows are selected when any of the distal pressures changes:

p4, p5, or p6
Training input

The distal model uses the same 8-dimensional input structure:

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Feature meaning
PX, PY = previous distal local tip features
cosP, sinP = distal bending-plane representation
dPX, dPY, dcosP, dsinP = change from previous step
Training target
[P4, P5, P6]
Saved processed CSV format
PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP, P4, P5, P6
Input/output shape
X.shape = [N, 8]
Y.shape = [N, 3]
Deployment input
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Deployment output
[P4, P5, P6]
6. Proximal-to-distal response model (P2D / crosstalk-related model)
Purpose

Model the distal tip response caused by proximal actuation.

Current implementation

In the current uploaded preprocessing script, this model is built as:

[P1, P2, P3] -> [X, Y]

This means the present P2D dataset is a pressure-to-distal-response mapping, rather than a full state-based crosstalk model.

Training input
[P1, P2, P3]
Training target
[X, Y]

where [X, Y] is the processed local distal 2D tip response.

Saved processed CSV format
P1, P2, P3, X, Y
Input/output shape
X.shape = [N, 3]
Y.shape = [N, 2]
Deployment input
[P1, P2, P3]
Deployment output
[X, Y]
7. Distal-to-proximal response model (D2P / crosstalk-related model, optional)
Purpose

Model the proximal tip response caused by distal actuation.

Note

A D2P preprocessing script is not included in the currently uploaded files.
If a symmetric reverse-direction crosstalk model is needed, the recommended dataset format is:

[P4, P5, P6] -> [X, Y]

where [X, Y] represents the processed local proximal 2D tip response.

Recommended training input
[P4, P5, P6]
Recommended training target
[X, Y]
Recommended saved processed CSV format
P4, P5, P6, X, Y
Recommended input/output shape
X.shape = [N, 3]
Y.shape = [N, 2]
Recommended deployment input
[P4, P5, P6]
Recommended deployment output
[X, Y]
8. RSCC pipeline requirement

The RSCC pipeline uses the segment-end states together with the trained models.

Minimum sensing requirement
one reference sensor
one proximal segment-end sensor
one distal segment-end sensor
Minimum model requirement

At minimum, the current logic supports:

proximal pressure mapping model

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P1, P2, P3]

distal pressure mapping model

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P4, P5, P6]

proximal-to-distal response model

[P1, P2, P3] -> [X, Y]

Optional:
4. distal-to-proximal response model

[P4, P5, P6] -> [X, Y]
State representation

For downstream control, the robot state is represented by the two segment-end tip states:

s_prox = proximal segment-end tip state
s_dist = distal segment-end tip state

These states are transformed into the required model input features before inference.# Sensor, Data, and Model Input Requirements

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
Notes
time is the timestamp column
p1 to p6 are chamber pressures
ks is the logged stiffness-related index
0A, 0B, and 0C are tracked sensor streams
each sensor must provide both position and quaternion orientation
keep the raw column names exactly as above so the current scripts can read them directly
Example raw row
time,p1,p2,p3,p4,p5,p6,ks,0A_pos_x,0A_pos_y,0A_pos_z,0A_orient_x,0A_orient_y,0A_orient_z,0A_orient_w,...
1762606570.41,0,0,0,0,0,0,0,15.3788,29.1995,-210.7228,...
2. Sensor placement requirement

To reproduce the current preprocessing pipeline, the raw log must contain:

one reference sensor
one proximal segment-end sensor
one distal segment-end sensor

In the current scripts, these are read from the three raw tracked streams 0A, 0B, and 0C, then internally transformed into local features.

For the downstream control pipeline, the important states are the two segment-end tip states:

proximal segment end
distal segment end
3. Preprocessing requirement

The models are not trained directly from all raw rows.

The preprocessing scripts first:

load the raw log
detect rows around pressure changes
shift the selected rows by -2 samples to account for valve delay
estimate zero-reference poses from zero-pressure regions
transform the raw sensor poses into local segment features
save processed datasets as .npz, normalization stats as .npz, and inspection tables as .csv
4. Proximal pressure mapping model
Purpose

Predict proximal chamber pressures [P1, P2, P3].

Raw data trigger

Rows are selected when any of the proximal pressures changes:

p1, p2, or p3
Training input

The current preprocessing script builds an 8-dimensional input vector:

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Feature meaning
PX, PY = previous proximal local tip features
cosP, sinP = proximal bending-plane representation
dPX, dPY, dcosP, dsinP = change from previous step
Training target
[P1, P2, P3]
Saved processed CSV format
PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP, P1, P2, P3
Input/output shape
X.shape = [N, 8]
Y.shape = [N, 3]
Deployment input
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Deployment output
[P1, P2, P3]
5. Distal pressure mapping model
Purpose

Predict distal chamber pressures [P4, P5, P6].

Raw data trigger

Rows are selected when any of the distal pressures changes:

p4, p5, or p6
Training input

The distal model uses the same 8-dimensional input structure:

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Feature meaning
PX, PY = previous distal local tip features
cosP, sinP = distal bending-plane representation
dPX, dPY, dcosP, dsinP = change from previous step
Training target
[P4, P5, P6]
Saved processed CSV format
PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP, P4, P5, P6
Input/output shape
X.shape = [N, 8]
Y.shape = [N, 3]
Deployment input
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP]
Deployment output
[P4, P5, P6]
6. Proximal-to-distal response model (P2D / crosstalk-related model)
Purpose

Model the distal tip response caused by proximal actuation.

Current implementation

In the current uploaded preprocessing script, this model is built as:

[P1, P2, P3] -> [X, Y]

This means the present P2D dataset is a pressure-to-distal-response mapping, rather than a full state-based crosstalk model.

Training input
[P1, P2, P3]
Training target
[X, Y]

where [X, Y] is the processed local distal 2D tip response.

Saved processed CSV format
P1, P2, P3, X, Y
Input/output shape
X.shape = [N, 3]
Y.shape = [N, 2]
Deployment input
[P1, P2, P3]
Deployment output
[X, Y]
7. Distal-to-proximal response model (D2P / crosstalk-related model, optional)
Purpose

Model the proximal tip response caused by distal actuation.

Note

A D2P preprocessing script is not included in the currently uploaded files.
If a symmetric reverse-direction crosstalk model is needed, the recommended dataset format is:

[P4, P5, P6] -> [X, Y]

where [X, Y] represents the processed local proximal 2D tip response.

Recommended training input
[P4, P5, P6]
Recommended training target
[X, Y]
Recommended saved processed CSV format
P4, P5, P6, X, Y
Recommended input/output shape
X.shape = [N, 3]
Y.shape = [N, 2]
Recommended deployment input
[P4, P5, P6]
Recommended deployment output
[X, Y]
8. RSCC pipeline requirement

The RSCC pipeline uses the segment-end states together with the trained models.

Minimum sensing requirement
one reference sensor
one proximal segment-end sensor
one distal segment-end sensor
Minimum model requirement

At minimum, the current logic supports:

proximal pressure mapping model

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P1, P2, P3]

distal pressure mapping model

[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P4, P5, P6]

proximal-to-distal response model

[P1, P2, P3] -> [X, Y]

Optional:
4. distal-to-proximal response model

[P4, P5, P6] -> [X, Y]
State representation

For downstream control, the robot state is represented by the two segment-end tip states:

s_prox = proximal segment-end tip state
s_dist = distal segment-end tip state

These states are transformed into the required model input features before inference.

## 5. RSCC Pipeline Requirements

The **RSCC pipeline** requires both trained models and structured input data for recursive compensation.

## 5.1 Required models
For a two-segment system, the minimum required models are:

1. **Segment 1 pressure mapping model**
2. **Segment 2 pressure mapping model**
3. **Segment 2 crosstalk compensation model**  
   predicts the effect of Segment 1 actuation on Segment 2

Depending on the implementation, additional crosstalk models may be included.

## 5.2 Required runtime input data
The minimum runtime input to RSCC is:

- **tip position at the end of Segment 1**
- **tip position at the end of Segment 2**
- desired target position for the segment being controlled
- previous or current pressure command
- any feature transformation required by the trained models

In other words, the pipeline assumes that the state of the robot can be described using the **two segment-end tip positions**.

## 5.3 RSCC data flow
A typical two-segment RSCC control flow is:

1. Measure current endpoint of Segment 1
2. Measure current endpoint of Segment 2
3. Predict pressure for Segment 1
4. Estimate crosstalk effect on Segment 2 caused by Segment 1
5. Compensate the target of Segment 2
6. Predict pressure for Segment 2 using compensated target
7. Send pressure commands to the robot

## 5.4 Minimum input representation for RSCC
A practical RSCC input representation is:

| Variable | Description |
|---|---|
| `s1 = [x1, y1, z1]` | Endpoint of Segment 1 |
| `s2 = [x2, y2, z2]` | Endpoint of Segment 2 |
| `s1_target` | Desired Segment 1 endpoint |
| `s2_target` | Desired Segment 2 endpoint |
| `p_prev` | Previous pressure command |
| `features(...)` | Derived features used by the trained models |

---

## 6. Data Quality Requirements

To ensure reliable model training and deployment, the dataset should satisfy the following conditions:

- synchronized sensor and pressure data
- consistent coordinate frame
- consistent units
- low missing-data rate
- sufficient coverage of the workspace
- sufficient variation in motion direction and bending magnitude
- repeated trials if hysteresis and drift are relevant
- clearly separated training, validation, and test datasets

Recommended:
- remove warm-up transients if needed
- document filtering and smoothing steps
- store normalization parameters used during training for deployment reuse

---

## 7. File Naming Recommendation

Recommended folder structure:

```text
data/
  raw/
    trial_001.xlsx
    trial_002.xlsx
  processed/
    train_segment1.csv
    train_segment2.csv
    train_crosstalk.csv
  deploy_examples/
    deploy_input_example.csv
models/
  segment1_pressure_model.pt
  segment2_pressure_model.pt
  segment2_crosstalk_model.pt

3) Prepare data (example for a P→D dataset)  
   python Data_sep/data_prepare_shift_frame_refA_results.py  
   Outputs land in data_sep/ as *.npz plus normalization stats *_norm.npz.
<img width="1813" height="845" alt="fig5" src="https://github.com/user-attachments/assets/ea4d862d-7aee-41c3-bb58-65a508600559" />

4) Train models (example: proximal IK)  
   python Train_code/train_seq_model_LSTM_P2.py  
   Checkpoints and loss logs save to model/. Similar scripts exist for distal IK, P→D crosstalk, and D→P crosstalk.

5) Generate RSCC pressures  
   python RSCC_pressuregen/test_model_with_GT_PD_shape_LSTM_11_cross_H.py  
   - Loads checkpoints from Trained_models/ by default; adjust model_name_* paths if you train new models.  
   - Writes commands to command/<model_name>shape_pressure_GT_1_7_1_5.csv and shows diagnostic pressure plots.

## Notes
- All paths are relative; the scripts expect data_sep/, model/, and command/ directories to exist or will create them.
- RSCC loop uses sequence_length=4, hidden_size=128, num_layers=2–3, dropout 0.1, and enforces the “minimum chamber = 0” rule per timestep.
- Example trajectories in test_model_with_GT_PD_shape_LSTM_11_cross_H.py generate clockwise circles for both segments; edit generate_circle_trajectory* to test other shapes.
- Add your raw ROS logs to Example_data/raw_data/ and update filenames in the data-prep scripts to rebuild datasets.
- Pretrained weights:  
  - Proximal IK: Trained_models/resultrealtime_log_ros_20251108_205609P.pth  
  - Distal IK: Trained_models/resultrealtime_log_ros_20251108_214407D.pth  
  - Crosstalk P→D: Trained_models/crosstalkP2D_4_128_LSTM.pth  
  - Crosstalk D→P: Trained_models/crosstalkD2P_4_128_LSTM.pth

## Citation
If you use this code, please cite the accompanying RAL 2026 submission (see "Data-Efficient Modeling of Hysteresis and Crosstalk for Inverse Kinematics of Soft Manipulators",).

Any further question plase contact email: k.borvorntajanya22@imperial.ac.uk
