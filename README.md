# RSCC Pressure Generation – Code for RAL 2026 submission

This repository contains data preparation, training, and inference scripts for the RAL 2025 paper “ML Hysteresis & Crosstalk Compensation for a Soft Manipulator.” It trains LSTM/Transformer models for inverse kinematics and crosstalk compensation, and generates RSCC pressure commands for two soft arm segments.

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

2) Prepare data (example for a P→D dataset)  
   python Data_sep/data_prepare_shift_frame_refA_results.py  
   Outputs land in data_sep/ as *.npz plus normalization stats *_norm.npz.
<img width="1813" height="845" alt="fig5" src="https://github.com/user-attachments/assets/ea4d862d-7aee-41c3-bb58-65a508600559" />

3) Train models (example: proximal IK)  
   python Train_code/train_seq_model_LSTM_P2.py  
   Checkpoints and loss logs save to model/. Similar scripts exist for distal IK, P→D crosstalk, and D→P crosstalk.

4) Generate RSCC pressures  
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
