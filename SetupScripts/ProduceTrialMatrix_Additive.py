# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:05:40 2025

@author: Sebastian Montesinos

Produces the sequence of trials for each participant, using an additive rule:
- Tail and Shape each contribute independently to total food.
- Color is present but does not affect food.
"""

import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# ============================
# --- Parameters ---
# ============================
mean_high = 3.5   # Means of distributions to draw from
mean_low = 1.5
std_dev = 1   # Spread of each distribution
participants = 30
save_csv = True
output_dir = r'C:\Users\Seb\Desktop\TestExp\Trial_Files' #Output path
output_dir_pilot = r'C:\Users\Seb\Desktop\P-A Scripts\Pilot-Files'
os.makedirs(output_dir, exist_ok=True)
training_reps = 3
testing_reps = 1
testing = 0 #Turn off testing for now (may not need it)


# ============================
# --- Visualize Distributions ---
# ============================
n_samples = 100000  # Number to sample

# Sample low distribution
low_samples = np.random.normal(mean_low, std_dev, n_samples)
low_samples = np.clip(np.round(low_samples), 0, 10).astype(int)

# Sample high distribution
high_samples = np.random.normal(mean_high, std_dev, n_samples)
high_samples = np.clip(np.round(high_samples), 0, 10).astype(int)

# Count occurrences
low_counts = np.bincount(low_samples, minlength=11)  
high_counts = np.bincount(high_samples, minlength=11) 
values = np.arange(0, 11)

# Plot histogram
plt.figure(figsize=(8, 4))
plt.bar(values - 0.2, low_counts / n_samples, width=0.4, alpha=0.7, label="Low")
plt.bar(values + 0.2, high_counts / n_samples, width=0.4, alpha=0.7, label="High")
plt.xticks(values)
plt.xlabel("Food value")
plt.ylabel("Probability")
plt.title("Food Distribution based on features")
plt.legend()
plt.tight_layout()
plt.show()



# ============================
# --- Stimulus Setup ---
# ============================
tails = ["T", "N"]   # Tail / No Tail
colors = ["B", "Y"]  # Blue / Yellow (does not affect food)
shapes = ["S", "C"]  # Square / Circle

stimuli = []
for t, c, s in itertools.product(tails, colors, shapes):
    stimuli.append({
        "tail": t,
        "color": c,
        "shape": s,
        "image_file": f"{t}_{c}_{s}.png"
    })

stim_df = pd.DataFrame(stimuli)

# ============================
# --- Sampling Function ---
# ============================
def sample_additive(tail, shape):
    """
    Sample food amount using an additive rule:
      - Each feature contributes independently.
      - Tail and Square add positively.
      - Each contribution is drawn from its respective distribution.
    """
    # Sample each feature independently
    tail_food = np.random.normal(mean_high if tail == "T" else mean_low, std_dev)
    shape_food = np.random.normal(mean_high if shape == "S" else mean_low, std_dev)

    # Clip each contribution so it can't be negative
    tail_food = np.clip(tail_food, 0, 10)
    shape_food = np.clip(shape_food, 0, 10)

    total = tail_food + shape_food

    # Round and ensure final value is between 1-10
    total_rounded = int(np.clip(round(total), 1, 10))

    # Print result (this is just to debug)
    print(f"tail={tail} ({tail_food:.2f}), shape={shape} ({shape_food:.2f}) → total={total:.2f} → rounded={total_rounded}")

    return total_rounded


# ============================
# --- Label Categories in dataframe ---
# ============================
def label_category(tail, shape):
    if tail == "T" and shape == "S":
        return "high"
    elif tail == "N" and shape == "C":
        return "low"
    else:
        return "medium"

# ============================
# --- Trial Generation ---
# ============================
def generate_trials(participant_id, training_reps=2, testing_reps=1):
    """
    Create a dataframe with randomized training and testing trials for one participant.
    """
    training_trials = []
    for i in range(training_reps):
        shuffled = stim_df.sample(frac=1).reset_index(drop=True)
        shuffled["phase"] = "training"
        shuffled["rep"] = i + 1
        shuffled["trial_num"] = range(1 + i * len(shuffled), 1 + (i + 1) * len(shuffled))
        shuffled["food_amount"] = [
            sample_additive(t, s) for t, s in zip(shuffled["tail"], shuffled["shape"])
        ]
        shuffled["category"] = [label_category(t, s) for t, s in zip(shuffled["tail"], shuffled["shape"])]
        training_trials.append(shuffled)

    training_df = pd.concat(training_trials, ignore_index=True)

    testing_trials = []
    start_trial = len(training_df)
    for i in range(testing_reps):
        shuffled_test = stim_df.sample(frac=1).reset_index(drop=True)
        shuffled_test["phase"] = "testing"
        shuffled_test["rep"] = i + 1
        shuffled_test["trial_num"] = range(start_trial + 1, start_trial + 1 + len(shuffled_test))
        shuffled_test["food_amount"] = [
            sample_additive(t, s) for t, s in zip(shuffled_test["tail"], shuffled_test["shape"])
        ]
        shuffled_test["category"] = [label_category(t, s) for t, s in zip(shuffled_test["tail"], shuffled_test["shape"])]
        start_trial += len(shuffled_test)
        testing_trials.append(shuffled_test)

    testing_df = pd.concat(testing_trials, ignore_index=True)
    all_trials = pd.concat([training_df, testing_df], ignore_index=True)
    all_trials["participant_id"] = participant_id
    all_trials["food_image_file"] = all_trials["food_amount"].astype(str) + "_food.png"
    return all_trials

# ============================
# --- Analysis Function ---
# ============================
def compute_feature_correlations(df):
    """Compute correlations of features and stimulus averages with food needed for training and testing phases."""
    
    participant = df['participant_id'].iloc[0]
    results = []
    for phase in ["training"]:
        df_phase = df[df['phase'] == phase].copy()
        df_phase["tail_num"] = df_phase["tail"].map({"T": 1, "N": 0})
        df_phase["shape_num"] = df_phase["shape"].map({"S": 1, "C": 0})
        df_phase["color_num"] = df_phase["color"].map({"B": 1, "Y": 0})

        corr_tail = df_phase["tail_num"].corr(df_phase["food_amount"])
        corr_shape = df_phase["shape_num"].corr(df_phase["food_amount"])
        corr_color = df_phase["color_num"].corr(df_phase["food_amount"])
        results.append({
        "participant": participant,
        "phase": phase,
        "corr_tail": corr_tail,
        "corr_shape": corr_shape,
        "corr_color": corr_color})

        print(f"\n{phase.capitalize()} phase correlations for participant {participant}:")
        print(f"  Tail (Has one):  {corr_tail:.3f}")
        print(f"  Shape (Square): {corr_shape:.3f}")
        print(f"  Color (Blue): {corr_color:.3f} (should be ~0)")

        stim_means = (
            df_phase.groupby("image_file", as_index=False)["food_amount"]
            .mean()
            .rename(columns={"food_amount": "mean_food_amount"})
        )
        print(f"\nAverage food amount per stimulus ({phase}):")
        for _, row in stim_means.iterrows():
            print(f"  {row['image_file']}: {row['mean_food_amount']:.2f}")
    df_corr = pd.DataFrame(results)
    return df_corr

# ============================
# --- Run Experiment Generation ---
# ============================
all_data = []
for participant_id in range(1, participants + 1):
    df = generate_trials(participant_id, training_reps=training_reps, testing_reps=testing_reps)
    all_data.append(df)
    correlations = correlations_training = compute_feature_correlations(df)
    out_path2 = os.path.join(output_dir_pilot, f"subj{participant_id:03d}_trials.csv")
    correlations.to_csv(out_path2, index = False)
        # Below adds the paths to all the images in a randomized order for testing
    fixed_images = [
        "Resources/T_B_S.png", "Resources/T_Y_S.png", "Resources/T_B_C.png", "Resources/T_Y_C.png",
        "Resources/N_B_S.png", "Resources/N_Y_S.png", "Resources/N_B_C.png", "Resources/N_Y_C.png"
    ]

    rand_images = np.random.permutation(fixed_images).tolist()
    df["images_list"] = ",".join(rand_images)


    if save_csv:
        # Save CSVs, only saving training for now
        out_path = os.path.join(output_dir, f"subj{participant_id:03d}_trials.csv")
        if testing:
            df.to_csv(out_path, index=False)
        df[df['phase'] == 'training'].to_csv(os.path.join(output_dir, f"subj{participant_id:03d}_training.csv"), index=False)
        if testing:
            df[df['phase'] == 'testing'].to_csv(os.path.join(output_dir, f"subj{participant_id:03d}_testing.csv"), index=False)

# Combine all participants’ data
if testing:
    all_data_df = pd.concat(all_data, ignore_index=True)
    all_data_df.to_csv(os.path.join(output_dir, "all_trials.csv"), index=False)

print("\nPreview:")
print(df[['phase', 'tail', 'shape', 'color', 'food_amount', 'category', 'trial_num']])
