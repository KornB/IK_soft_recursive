import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import ast
import math
import os
file_text = 'realtime_log_ros_20251108_205609P'
file_name = file_text+'.csv'
file_zero = 'realtime_log_ros_20251108_205609P.csv'
file_save = 'result'+file_text+'2.npz'
file_save_norm = 'result'+file_text+'_norm.npz'
file_save_csv = 'result'+file_text+'.csv'
def pose_to_matrix(pos, quat):
    r = R.from_quat([quat[0], quat[1], quat[2], quat[3]])  # [x, y, z, w]
    T = np.eye(4)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = pos
    return T

def matrix_to_pose(T):
    pos = T[:3, 3]
    r = R.from_matrix(T[:3, :3])
    quat = r.as_quat()  # [x, y, z, w]
    return pos, [quat[3], quat[0], quat[1], quat[2]]  # [w, x, y, z]

def compute_bending_features_yz(pos_yz):
    norm = np.linalg.norm(pos_yz)
    if norm == 0:
        return 1.0, 0.0
    phi = math.atan2(pos_yz[1], pos_yz[0])  # atan2(Z, Y)
    return math.cos(phi), math.sin(phi)
import pandas as pd
import ast

def print_last_zero_pressure_positions(csv_file):
    # 1) Load and parse the 'pressure' column into real lists
    df = pd.read_csv(csv_file)
    zero_mask = (df[['p1','p2','p3']].fillna(0) == 0).all(axis=1)

    # Find transitions
    last_zero_idxs = [
        i-2 for i in zero_mask.index
        if zero_mask.iloc[i] and (i + 1 < len(zero_mask) and not zero_mask.iloc[i+1])
    ]

    # 3) Find indices i where zero_mask[i] is True but zero_mask[i+1] is False
    last_zero_idxs = [
        i-2 for i in zero_mask.index
        if zero_mask.iloc[i] and (i + 1 < len(zero_mask) and not zero_mask.iloc[i+1])
    ] #some delay on valve

    if not last_zero_idxs:
        print("No zero→nonzero transitions found.")
        return

    # 4) Extract position & quaternion for 0B and 0C
    pos0A = df.loc[last_zero_idxs, ['0B_pos_x','0B_pos_y','0B_pos_z']].astype(float)
    quat0A = df.loc[last_zero_idxs, ['0B_orient_x','0B_orient_y','0B_orient_z','0B_orient_w']].astype(float)
    pos0C = df.loc[last_zero_idxs, ['0A_pos_x','0A_pos_y','0A_pos_z']].astype(float)
    quat0C = df.loc[last_zero_idxs, ['0A_orient_x','0A_orient_y','0A_orient_z','0A_orient_w']].astype(float)
    pos0B = df.loc[last_zero_idxs, ['0C_pos_x','0C_pos_y','0C_pos_z']].astype(float)
    quat0B = df.loc[last_zero_idxs, ['0C_orient_x','0C_orient_y','0C_orient_z','0C_orient_w']].astype(float)

    # # 5) Print each occurrence
    print("=== Sensor 0B at last-zero-before-change ===")
    for idx in last_zero_idxs:
        print(f"Row {idx}: pos=({pos0B.loc[idx].values}), quat=({quat0B.loc[idx].values})")
    print("\n=== Sensor 0C at last-zero-before-change ===")
    # for idx in last_zero_idxs:
        # print(f"Row {idx}: pos=({pos0C.loc[idx].values}), quat=({quat0C.loc[idx].values})")


    avg_pos0A = pos0A.mean().values
    avg_quat0A = quat0A.mean().values
    # 6) Compute & print averages
    avg_pos0B = pos0B.mean().values
    avg_quat0B = quat0B.mean().values
    avg_pos0C = pos0C.mean().values
    avg_quat0C = quat0C.mean().values

    print(f"\nAverage 0B pos: x={avg_pos0B[0]:.3f}, y={avg_pos0B[1]:.3f}, z={avg_pos0B[2]:.3f}")
    print(f"Average 0B quat: x={avg_quat0B[0]:.3f}, y={avg_quat0B[1]:.3f}, z={avg_quat0B[2]:.3f}, w={avg_quat0B[3]:.3f}")

    print(f"\nAverage 0C pos: x={avg_pos0C[0]:.3f}, y={avg_pos0C[1]:.3f}, z={avg_pos0C[2]:.3f}")
    print(f"Average 0C quat: x={avg_quat0C[0]:.3f}, y={avg_quat0C[1]:.3f}, z={avg_quat0C[2]:.3f}, w={avg_quat0C[3]:.3f}")
    return avg_pos0A, avg_quat0A,avg_pos0B, avg_quat0B,avg_pos0C, avg_quat0C


def build_shifted_YZ_features_dataset(
        csv_file, output_npz, norm_npz,
        start_pos_0A,start_pos_0B, start_pos_0C,
        start_quat_0A=None,start_quat_0B=None,start_quat_0C=None):
    df = pd.read_csv(csv_file)
    mask = (
    (df["p1"] != df["p1"].shift(-1)) |
    (df["p2"] != df["p2"].shift(-1)) |
    (df["p3"] != df["p3"].shift(-1)) |
    (df.index == df.index[-1])
    )
    mask = mask.shift(-2).fillna(False)
    df_last = df[mask].copy().reset_index(drop=True)
    # print(df_last)
    inputs = []
    targets = []
    prev_features = None
    T_0A_global_zero = pose_to_matrix(start_pos_0A,start_quat_0A)
    T_0B_global_zero = pose_to_matrix(start_pos_0B,start_quat_0B)
    T_0C_global_zero = pose_to_matrix(start_pos_0C,start_quat_0C)
    T_0B_local_zero = np.linalg.inv(T_0A_global_zero) @ T_0B_global_zero
    T_0C_local_zero = np.linalg.inv(T_0C_global_zero) @ T_0B_global_zero
    R_P_zero      = T_0B_local_zero[:3,:3]
    R_P_inv   = R_P_zero.T               # for rotation matrices, inv(R)=Rᵀ
    R_D_zero = T_0C_local_zero[:3,:3]
    R_D_inv = R_D_zero.T
    relP_pos_zero, _ = matrix_to_pose(T_0B_local_zero)
    relD_pos_zero, _ = matrix_to_pose(T_0C_local_zero)
    start_relD = np.array([relD_pos_zero[0], relD_pos_zero[1],relD_pos_zero[2]])
    start_relP = np.array([relP_pos_zero[0], relP_pos_zero[1], relP_pos_zero[2]])
    #B is distal C is proximal
    for i, row in df_last.iterrows():
        try:
            # print(row)
            # Skip if 0C has missing or invalid data
            # if any(pd.isna(row[f'0C_pos_{axis}']) or row[f'0C_pos_{axis}'] == '' for axis in ['x', 'y']):
                # print(f"Skipping index {i} due to missing 0C data")
                # continue
            p_tar = np.array([row['p1'], row['p2'], row['p3']], dtype=np.float64)
            ks = np.array([row['ks']], dtype=np.float64)
            # print(ks)
            pos_0C = np.array([row['0A_pos_x'], row['0A_pos_y'], row['0A_pos_z']], dtype=np.float64)
            quat_0C = np.array([row['0A_orient_x'], row['0A_orient_y'], row['0A_orient_z'], row['0A_orient_w']]) # w x y z (in the data is x y z w wrong)
            # print('hi')
            pos_0A = np.array([row['0B_pos_x'], row['0B_pos_y'], row['0B_pos_z']], dtype=np.float64)
            quat_0A = np.array([row['0B_orient_x'], row['0B_orient_y'], row['0B_orient_z'], row['0B_orient_w']]) # w x y z (in the data is x y z w wrong)

            pos_0B = np.array([row['0C_pos_x'], row['0C_pos_y'], row['0C_pos_z']], dtype=np.float64)
            quat_0B = np.array([row['0C_orient_x'], row['0C_orient_y'], row['0C_orient_z'], row['0C_orient_w']])

            if np.isnan(pos_0B).any():
                continue
            # print('here')
            T_0A_global = pose_to_matrix(pos_0A, quat_0A)
            T_0B_global = pose_to_matrix(pos_0B, quat_0B) #proximal
            T_0C_global = pose_to_matrix(pos_0C, quat_0C) #distal
            T_0B_local = np.linalg.inv(T_0A_global_zero) @ T_0B_global 
            T_0C_local = np.linalg.inv(T_0A_global_zero) @ T_0C_global #distal local
            relP_pos, _ = matrix_to_pose(T_0B_local)
            relD_pos, _ = matrix_to_pose(T_0C_local)
            relD_xy = np.array([relD_pos[0], relD_pos[1], relD_pos[2]])  # X, Y
            relD_xy = relD_xy-start_relD
            relD_xy = R_D_inv @ relD_xy      # now both centered and de‐tilted
            relP_xz = np.array([relP_pos[0], relP_pos[1], relP_pos[2]])
            relP_xz = relP_xz-start_relP
            relP_xz = R_P_inv @ relP_xz       # now both centered and de‐tilted
            # print(relP_xz)
            cosD_phi, sinD_phi = compute_bending_features_yz(relD_xy)
            cosP_phi, sinP_phi = compute_bending_features_yz(relP_xz)
            # current_features = np.concatenate([relP_xz, [cosP_phi, sinP_phi]])
            current_features = np.concatenate([relP_xz[0:2],[cosP_phi,sinP_phi]])
            # print('hi3')
            # current_features = np.concatenate([relP_xz,[cosP_phi,sinP_phi],relD_xy,[cosD_phi,sinD_phi]])
            if prev_features is None:
                prev_features = current_features
                prev_relP_xz= relP_xz
                continue
            # now is proximal [X dx Dxy] -> P[123]
            # if distal [X dx] ->P[456]
            # print('here2')
            delta = current_features - prev_features
            # print(ks)
            input_vector = np.concatenate([prev_features, delta])
            # print('hi2')
            # input_vector = np.concatenate([prev_features, delta,prev_relD_xy])
            inputs.append(input_vector)
            targets.append(p_tar)
            # prev_relD_xy = relD_xy
            prev_relP_xz = relP_xz
            prev_features = current_features

        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")
            continue

    inputs = np.array(inputs)
    targets = np.array(targets)
    print(inputs[0])
    np.savez(output_npz, inputs=inputs, targets=targets)
    print(f"Saved data to {output_npz}: {inputs.shape[0]} samples.")

    input_mean = np.mean(inputs, axis=0)
    input_std = np.std(inputs, axis=0)
    target_mean = np.mean(targets, axis=0)
    target_std = np.std(targets, axis=0)
        # Save to CSV for inspection
    df_inputs = pd.DataFrame(inputs, columns=['PX','PY','cosP','sinP','dPX','dPY','dcosP','dsinP'])
    df_targets = pd.DataFrame(targets, columns=['P1','P2','P3'])
    df_combined = pd.concat([df_inputs, df_targets], axis=1)

    csvpath = os.path.join(save_dir,file_save_csv)
    df_combined.to_csv(csvpath, index=False)
    print("Saved inputs and targets to inputs_targets_datasetD.csv")
    np.savez(norm_npz, input_mean=input_mean, input_std=input_std,
             target_mean=target_mean, target_std=target_std)
    print(f"Saved normalization stats to {norm_npz}")

if __name__ == "__main__":
    output_dir = 'exp_data'
    save_dir = 'data_sep'
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, file_name)
    zeropath = os.path.join(output_dir, file_zero)
    savepath = os.path.join(save_dir,file_save)
    savepath_norm = os.path.join(save_dir,file_save_norm)
    avg_p0A, avg_q0A,avg_p0B, avg_q0B,avg_p0C, avg_q0C = print_last_zero_pressure_positions(zeropath)

    # 2) build your dataset using those averages
    build_shifted_YZ_features_dataset(
        filepath, savepath, savepath_norm,
        start_pos_0A=avg_p0A,
        start_pos_0B=avg_p0B,
        start_pos_0C=avg_p0C,
        start_quat_0A=avg_q0A,
        start_quat_0B=avg_q0B,
        start_quat_0C=avg_q0C
    )
    # build_shifted_YZ_features_dataset(
    #     filepath,#realtime_log_2segments2_data
    #     savepath,
    #     savepath_norm
    # )
    # print_last_zero_pressure_positions(filepath)
