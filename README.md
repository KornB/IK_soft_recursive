<img width="1407" height="709" alt="image" src="https://github.com/user-attachments/assets/1bf3f3ab-32a1-4c85-9b6d-b8379d15a050" /># RSCC Pressure Generation – Code for RAL 2026 submission

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

  <img width="1407" height="709" alt="image" src="https://github.com/user-attachments/assets/270543b6-fcc5-48f0-9cb4-3bcbcd5f5a67" />


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

## 2. Raw Sensor Data File Format

Raw recorded data should be stored in `.xlsx` or `.csv` format. Each row should correspond to one synchronized time sample.

## 2.1 Minimum required columns
For a two-segment system, the raw file should contain:

| Column name | Description |
|---|---|
| `t` | Timestamp |
| `seg1_x` | X position of Segment 1 endpoint |
| `seg1_y` | Y position of Segment 1 endpoint |
| `seg1_z` | Z position of Segment 1 endpoint |
| `seg2_x` | X position of Segment 2 endpoint |
| `seg2_y` | Y position of Segment 2 endpoint |
| `seg2_z` | Z position of Segment 2 endpoint |
| `p1` | Chamber pressure 1 |
| `p2` | Chamber pressure 2 |
| `p3` | Chamber pressure 3 |
| `p4` | Chamber pressure 4 |
| `p5` | Chamber pressure 5 |
| `p6` | Chamber pressure 6 |

If the robot has a different number of chambers, replace the pressure columns accordingly.

## 2.2 Example raw data table

| t | seg1_x | seg1_y | seg1_z | seg2_x | seg2_y | seg2_z | p1 | p2 | p3 | p4 | p5 | p6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 1.2 | 0.4 | 45.1 | 2.8 | 0.7 | 88.5 | 20 | 18 | 25 | 30 | 28 | 31 |
| 0.10 | 1.4 | 0.5 | 45.4 | 3.0 | 0.8 | 88.9 | 21 | 18 | 24 | 30 | 29 | 31 |

---

## 3. Training Input Format

The raw sensor data are not fed directly into the models. They must first be converted into model-specific training inputs.

---

## 3.1 Pressure Mapping Model

### Purpose
This model predicts the pressure command of a segment from its state or target motion features.

### Typical training input
The exact feature set may vary by experiment, but the input generally contains:
- previous tip state
- previous pressure
- current or target tip state
- state difference terms

A typical input vector for one segment may include:

| Feature | Description |
|---|---|
| `x_(t-1), y_(t-1), z_(t-1)` | Previous tip position |
| `x_t, y_t, z_t` | Current or target tip position |
| `dx, dy, dz` | Position increment |
| `p_(t-1)` | Previous pressure command |
| optional bending features | e.g. `theta`, `phi`, `dtheta`, `dphi`, `cos(phi)`, `sin(phi)` |

### Typical training target
| Target | Description |
|---|---|
| `p_t` | Pressure command at current time step |

### Example processed training row
| x_prev | y_prev | z_prev | x_tgt | y_tgt | z_tgt | dx | dy | dz | p1_prev | p2_prev | p3_prev | p1_t | p2_t | p3_t |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.2 | 0.4 | 45.1 | 1.4 | 0.5 | 45.4 | 0.2 | 0.1 | 0.3 | 20 | 18 | 25 | 21 | 18 | 24 |

If using sequence models such as LSTM or Transformer, multiple consecutive rows are stacked into a window of length `m`.

### Sequence input example
For time window length `m`, the training input shape is typically:

- `X.shape = [N, m, F]`
- `Y.shape = [N, O]` or `[N, m, O]`

where:
- `N` = number of training samples
- `m` = sequence length
- `F` = number of input features
- `O` = number of output pressure channels

---

## 3.2 Crosstalk Model

### Purpose
This model predicts the offset or disturbance on one segment caused by the actuation or deformation of another segment.

### Typical training input
The input usually contains:
- state of the influencing segment
- state of the affected segment
- optionally pressure of the influencing segment
- optionally relative geometric features

A typical crosstalk input may include:
- proximal segment endpoint position
- distal segment endpoint position
- commanded pressure of the proximal segment
- relative displacement between segment endpoints

### Typical training target
The target is the crosstalk offset, for example:
- position offset of the distal endpoint
- equivalent correction term for the inverse model
- compensated target shift

### Example processed features
| Feature | Description |
|---|---|
| `seg1_x, seg1_y, seg1_z` | Endpoint of Segment 1 |
| `seg2_x, seg2_y, seg2_z` | Endpoint of Segment 2 |
| `rel_x, rel_y, rel_z` | Relative position |
| `p_seg1` | Pressure applied to Segment 1 |
| `eta` or `delta_x, delta_y, delta_z` | Crosstalk target |

---

## 4. Deployment Input Format

During deployment, the input format is different from offline training because only currently available measurements and desired target states can be used.

---

## 4.1 Pressure Mapping Model for deployment

### Required inputs
At each control step, the pressure mapping model typically requires:
- previous measured or estimated tip state
- desired next tip state
- previous pressure command
- derived motion features

### Minimal deployment input example
| Input | Description |
|---|---|
| `x_prev, y_prev, z_prev` | Previous measured tip position |
| `x_des, y_des, z_des` | Desired next tip position |
| `dx, dy, dz` | Desired increment |
| `p_prev` | Previous commanded pressure |

### Output
| Output | Description |
|---|---|
| `p_cmd` | New chamber pressure command |

---

## 4.2 Crosstalk Model for deployment

### Required inputs
The crosstalk model requires the currently available state of the segment that induces coupling, and the current or desired state of the affected segment.

### Minimal deployment input example
| Input | Description |
|---|---|
| `seg1_tip` | Current endpoint of Segment 1 |
| `seg2_tip` | Current or desired endpoint of Segment 2 |
| `p_seg1` | Current or next pressure of Segment 1 |

### Output
| Output | Description |
|---|---|
| `crosstalk_offset` | Predicted offset caused by inter-segment coupling |

This output is then used to compensate the target passed to the downstream segment model.

---

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
