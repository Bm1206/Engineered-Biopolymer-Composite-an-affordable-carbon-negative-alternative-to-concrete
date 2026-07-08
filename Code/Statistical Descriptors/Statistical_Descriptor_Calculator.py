# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 09:11:08 2026

@author: Barney
"""

import numpy as np
import tifffile
import pandas as pd
import time

voxel_size = 1.71  # micrometers per voxel

# ---------------- DESCRIPTORS ---------------- #

def lineal_path_function(image_slice, phase_value):
    shape = image_slice.shape
    max_dist = min(shape) // 2
    distances = np.arange(1, max_dist)
    L_r = []

    phase_mask = (image_slice == phase_value)

    for r in distances:
        count = 0
        total = 0

        for row in range(shape[0]):
            for col in range(shape[1] - r):
                if np.all(phase_mask[row, col:col+r]):
                    count += 1
                total += 1

        for col in range(shape[1]):
            for row in range(shape[0] - r):
                if np.all(phase_mask[row:row+r, col]):
                    count += 1
                total += 1

        L_r.append(count / total if total > 0 else 0)

    return distances * voxel_size, np.array(L_r)


def two_point_probability_pair(image_slice, phase1, phase2):
    shape = image_slice.shape
    max_dist = shape[1] // 2
    distances = np.arange(1, max_dist)
    S2_r = []

    for r in distances:
        count = 0
        total = 0

        for row in range(shape[0]):
            for col in range(shape[1] - r):
                if image_slice[row, col] == phase1 and image_slice[row, col + r] == phase2:
                    count += 1
                total += 1

        S2_r.append(count / total if total > 0 else 0)

    return distances * voxel_size, np.array(S2_r)

# ---------------- LOADER ---------------- #

def load_remapped_image(tif_path):
    """
    Loads a remapped image with values {0, 151, 255}
    """
    image = tifffile.imread(tif_path).astype(np.uint8)
    print(f"Loaded remapped image: {tif_path}, shape={image.shape}")

    # Safety check
    unique_vals = np.unique(image)
    print("Unique values:", unique_vals)

    return image

# ---------------- DATASETS ---------------- #
# Please load in the image set you would like to calculate the statistical 
# descriptors for here in the next two lines.

datasets = {
    "30 MPa data-1": {
        "path": "30_MPa_output_3d_256x256_comb_remapped_chunk_001.tif"
    }
}

phases = {
    "Aggregate": 255,
    "Binder": 151,
    "Void": 0
}

pairs = {
    "Aggregate_Aggregate": (255, 255),
    "Aggregate_Binder": (255, 151),
    "Void_Void": (0, 0)
}

# ---------------- MAIN LOOP ---------------- #

for name, info in datasets.items():
    print(f"\n--- Processing {name} ---")

    remapped_stack = load_remapped_image(info["path"])
    num_slices = remapped_stack.shape[0]

    rows = []

    # Distances (once per dataset)
    slice0 = remapped_stack[0]
    dist_L, _ = lineal_path_function(slice0, 255)
    dist_S2, _ = two_point_probability_pair(slice0, 255, 255)

    pd.DataFrame({
        "distance_lineal_path_um": dist_L,
        "distance_two_point_um": dist_S2
    }).to_csv(f"{name}_distances.csv", index=False)

    # Slice selection
    slice_indices = list(range(0, num_slices, 1))
    if (num_slices - 1) not in slice_indices:
        slice_indices.append(num_slices - 1)

    for i in slice_indices:
        print(f"Processing slice {i+1}/{num_slices}... ", end="")
        start = time.time()

        slice_img = remapped_stack[i]
        row_data = {"slice_index": i}

        for phase_name, phase_val in phases.items():
            _, L_vals = lineal_path_function(slice_img, phase_val)
            for j, val in enumerate(L_vals):
                row_data[f"L_{phase_name}_{dist_L[j]:.2f}um"] = val

        for pair_name, (p1, p2) in pairs.items():
            _, S2_vals = two_point_probability_pair(slice_img, p1, p2)
            for j, val in enumerate(S2_vals):
                row_data[f"S2_{pair_name}_{dist_S2[j]:.2f}um"] = val

        row_data["processing_time_sec"] = time.time() - start
        print("done")

        rows.append(row_data)

    df = pd.DataFrame(rows)
    df.to_csv(f"{name}_all_slices_descriptors.csv", index=False)
    print(f"Saved descriptors for {name}")
