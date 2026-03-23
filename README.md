# IK Soft Recursive

Code and example data for recursive inverse kinematics and crosstalk compensation in a two-segment soft manipulator.

This repository contains preprocessing, training, and inference code for segment-wise pressure prediction and recursive segment-wise crosstalk compensation (RSCC). The current codebase includes:
- proximal pressure mapping
- distal pressure mapping
- proximal-to-distal crosstalk-related modeling
- RSCC pressure generation for two-segment execution

---

## Overview

Soft continuum manipulators with morphable chambers exhibit hysteresis and inter-segment coupling.  
This repository implements a recursive pipeline that combines:
- **segment-wise inverse pressure models**
- **crosstalk-related compensation models**
- **iterative pressure generation for multi-segment control**

The repository is organized to support:
1. raw log preprocessing
2. model training
3. pretrained model loading
4. RSCC pressure generation and testing

For detailed paper notes, see [`docs/paper_details.md`](docs/paper_details.md).  
For detailed raw-data and processed-data format, see [`docs/data_format.md`](docs/data_format.md).

---

## Repository Layout

| Path | Description |
|---|---|
| `environment.yml` | Conda environment for preprocessing, training, and inference |
| `Data_sep/` | Dataset builders that convert raw logs into processed datasets |
| `Train_code/` | Training scripts for proximal, distal, and crosstalk-related models |
| `RSCC_pressuregen/` | RSCC inference / pressure generation scripts |
| `Trained_models/` | Pretrained model checkpoints |
| `Example_data/raw_data/` | Example raw sensor logs |
| `Example_data/process_data/` | Example processed datasets |
| `docs/paper_details.md` | Detailed summary of the paper and how the repository maps to it |
| `docs/data_format.md` | Detailed raw-data and processed-data specification |
| `RAL_2025_ML_Hysteresis_Crosstalk_SoftMani_Re1_submitted_main.pdf` | Paper PDF |

---

## Quick Start

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate torchenv
```

### 2. Prepare data

Run the preprocessing scripts in `Data_sep/` to convert raw CSV logs into processed datasets.

Typical outputs include:
- processed training data in `.npz`
- normalization statistics in `.npz`
- inspection tables in `.csv`

### 3. Train models

Run the training scripts in `Train_code/` to train:
- proximal pressure mapping model
- distal pressure mapping model
- crosstalk-related models

Trained models are typically saved as `.pth` checkpoints.

### 4. Run RSCC pressure generation

Use the scripts in `RSCC_pressuregen/` together with trained `.pth` models to generate pressure commands for two-segment control.

---

## End-to-End Workflow

The typical workflow in this repository is:

```text
Example_data/raw_data/*.csv
    ↓
Data_sep/ (preprocessing)
    ↓
Example_data/process_data/*.npz / *.csv
    ↓
Train_code/ (model training)
    ↓
Trained_models/*.pth
    ↓
RSCC_pressuregen/ (inference / pressure generation)
```

---

## Step-by-Step Commands

### 1. Preprocessing (raw → processed data)

Convert raw sensor logs into model-ready datasets.

Example command:

```bash
python Data_sep/data_prepare_shift_frame_1seg_8_10_P.py --log realtime_log_ros_20251108_205609P.csv
```

Input:

```text
Example_data/raw_data/realtime_log_ros_20251108_205609P.csv
```

Output:

```text
Example_data/process_data/result*.npz
Example_data/process_data/result*_norm.npz
Example_data/process_data/result*.csv
```

### 2. Train Models (processed → `.pth`)

Example commands:

```bash
python Train_code/train_proximal_LSTM.py
python Train_code/train_distal_LSTM.py
python Train_code/train_crosstalk_P2D.py
python Train_code/train_crosstalk_D2P.py
```

Output:

```text
Trained_models/*.pth
```

> Replace the training script names above with the actual filenames in `Train_code/` if they differ in your local repository.

### 3. Inference / Pressure Generation

#### (A) Proximal only

```bash
python RSCC_pressuregen/predict_proximal_only.py
```

#### (B) Distal only

```bash
python RSCC_pressuregen/predict_distal_only.py
```

#### (C) Crosstalk only

```bash
python RSCC_pressuregen/predict_crosstalk_P2D_only.py
python RSCC_pressuregen/predict_crosstalk_D2P_only.py
```

#### (D) Full RSCC

```bash
python RSCC_pressuregen/rscc_pressure_generation.py
```

Output:

```text
Command/*.csv
```

> Replace the inference script names above with the actual filenames in `RSCC_pressuregen/` if they differ in your local repository.

---

## RSCC Control Pipeline

```mermaid
flowchart TD
    A[Raw CSV: 0A, 0B, 0C] --> B[Preprocessing<br/>frame transform + feature extraction]
    B --> C[Segment States<br/>s_prox, s_dist]
    C --> D[Proximal Model<br/>P1, P2, P3]
    D --> E[P2D Crosstalk<br/>distal offset]
    E --> F[Compensated Distal Target]
    F --> G[Distal Model<br/>P4, P5, P6]
    G --> H[D2P Crosstalk<br/>proximal offset]
    H --> I[Compensated Proximal Target]
    I --> C
```

The RSCC pipeline performs recursive compensation between the proximal and distal segments.  
Initial pressure commands are estimated by the segment-wise models, crosstalk effects are predicted, and the segment states are updated iteratively.

---

## Model Summary

### Proximal pressure model

```text
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P1, P2, P3]
```

### Distal pressure model

```text
[PX, PY, cosP, sinP, dPX, dPY, dcosP, dsinP] -> [P4, P5, P6]
```

### P2D response model

```text
[P1, P2, P3] -> [X, Y]
```

### D2P response model

```text
[P4, P5, P6] -> [X, Y]
```

For the full raw-data format, processed dataset format, and model I/O details, see [`docs/data_format.md`](docs/data_format.md).

---

## Suggested Usage Order

For a new user of the repository, the recommended order is:

1. Read this `README.md`
2. Check [`docs/paper_details.md`](docs/paper_details.md)
3. Check [`docs/data_format.md`](docs/data_format.md)
4. Start from `Example_data/raw_data/`
5. Run preprocessing in `Data_sep/`
6. Train or load models from `Train_code/` and `Trained_models/`
7. Run inference in `RSCC_pressuregen/`

---

## Citation

If you use this repository, please cite both the repository and the accompanying paper.

### Repository citation

A `CITATION.cff` file is provided in the repository root so GitHub can display a citation entry automatically.

### Paper citation

```bibtex
@misc{korn_soft_recursive,
  title        = {Data-Efficient Modeling of Hysteresis and Crosstalk for Inverse Kinematics of Soft Continuum Robots},
  author       = {Korn Borvornatanajanya and collaborators},
  year         = {2026},
  note         = {RAL submission / manuscript included in repository},
}
```

Update this BibTeX entry once the final publication details, pages, and DOI are available.

---

## Paper and Documentation

- Paper PDF: `RAL_2025_ML_Hysteresis_Crosstalk_SoftMani_Re1_submitted_main.pdf`
- Detailed paper notes: [`docs/paper_details.md`](docs/paper_details.md)
- Detailed data specification: [`docs/data_format.md`](docs/data_format.md)

---

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

---

## Contact

For questions related to the repository, models, or paper, please open an issue or contact the repository author.
k.borvorntajanya22@imperial.ac.uk
korn.bov@gmail.com
