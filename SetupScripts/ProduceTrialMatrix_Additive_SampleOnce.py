# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 08:53:54 2025

@author: Seb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 11:05:40 2025

@author: Seb
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:48:43 2025
Modified: Oct 28, 2025
@author: Seb

Produce trialmatrixes for experiment using the following rules:
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
participants = 2
save_csv = True
output_dir = r'C:\Users\Seb\Desktop\TestExp\Trial_Files'
os.makedirs(output_dir, exist_ok=True)
training_reps = 3
testing_reps = 1
testing = 0 #Turn off testing for now (may not need it)


# Define category distributions
mean_low, mean_med, mean_high = 3, 5, 8
std_dev = 1.5
n_samples = 100000

# Sample from each category
low_samples = np.random.normal(mean_low, std_dev, n_samples)
med_samples = np.random.normal(mean_med, std_dev, n_samples)
high_samples = np.random.normal(mean_high, std_dev, n_samples)

# Clip to 1–10 range
low_samples = np.clip(np.round(low_samples), 1, 10).astype(int)
med_samples = np.clip(np.round(med_samples), 1, 10).astype(int)
high_samples = np.clip(np.round(high_samples), 1, 10).astype(int)

# Count occurrences
low_counts = np.bincount(low_samples, minlength=11)
med_counts = np.bincount(med_samples, minlength=11)
high_counts = np.bincount(high_samples, minlength=11)
values = np.arange(0, 11)

# Plot all three distributions
plt.figure(figsize=(9, 5))
plt.bar(values - 0.25, low_counts / n_samples, width=0.25, alpha=0.7, label="Low (mean≈2)")
plt.bar(values, med_counts / n_samples, width=0.25, alpha=0.7, label="Medium (mean≈5)")
plt.bar(values + 0.25, high_counts / n_samples, width=0.25, alpha=0.7, label="High (mean≈8)")
plt.xticks(values)
plt.xlabel("Food value")
plt.ylabel("Probability")
plt.title("Food Value Distributions for Low, Medium, and High Categories")
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
# --- Sampling Food ---
# ============================
def sample_additive(tail, shape):
    """
    Assign food amount based on feature combination:
      - low, medium, high categories each have their own distribution
      - sample once from that distribution
    """
    # Assign category based on features
    if tail == "T" and shape == "S":
        category = "high"
    elif tail == "N" and shape == "C":
        category = "low"
    else:
        category = "medium"
    
    # Sample based on category
    if category == "low":
        food = np.random.normal(loc=mean_low, scale=std_dev)
    elif category == "medium":
        food = np.random.normal(loc=mean_med, scale=std_dev)
    else:  # high
        food = np.random.normal(loc=mean_high, scale=std_dev)
    
    # Round and clamp between 1–10
    food_rounded = int(np.clip(round(food), 1, 10))
    
    # Debug print
    print(f"tail={tail}, shape={shape}, category={category} → food={food:.2f} → rounded={food_rounded}")
    
    return food_rounded, category


# ============================
# --- Category Labeling  ---
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
        shuffled["food_amount"], shuffled["category"] = zip(
            *[sample_additive(t, s) for t, s in zip(shuffled["tail"], shuffled["shape"])]
        )

        training_trials.append(shuffled)

    training_df = pd.concat(training_trials, ignore_index=True)

    testing_trials = []
    start_trial = len(training_df)
    for i in range(testing_reps):
        shuffled_test = stim_df.sample(frac=1).reset_index(drop=True)
        shuffled_test["phase"] = "testing"
        shuffled_test["rep"] = i + 1
        shuffled_test["trial_num"] = range(start_trial + 1, start_trial + 1 + len(shuffled_test))
        shuffled_test["food_amount"], shuffled_test["category"] = zip(
            *[sample_additive(t, s) for t, s in zip(shuffled_test["tail"], shuffled_test["shape"])]
        )
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

    for phase in ["training", "testing"]:
        df_phase = df[df['phase'] == phase].copy()
        df_phase["tail_num"] = df_phase["tail"].map({"T": 1, "N": 0})
        df_phase["shape_num"] = df_phase["shape"].map({"S": 1, "C": 0})
        df_phase["color_num"] = df_phase["color"].map({"B": 1, "Y": 0})

        corr_tail = df_phase["tail_num"].corr(df_phase["food_amount"])
        corr_shape = df_phase["shape_num"].corr(df_phase["food_amount"])
        corr_color = df_phase["color_num"].corr(df_phase["food_amount"])

        print(f"\n{phase.capitalize()} phase correlations for participant {participant}:")
        print(f"  Tail:  {corr_tail:.3f}")
        print(f"  Shape: {corr_shape:.3f}")
        print(f"  Color: {corr_color:.3f} (should be ~0)")

        stim_means = (
            df_phase.groupby("image_file", as_index=False)["food_amount"]
            .mean()
            .rename(columns={"food_amount": "mean_food_amount"})
        )
        print(f"\nAverage food amount per stimulus ({phase}):")
        for _, row in stim_means.iterrows():
            print(f"  {row['image_file']}: {row['mean_food_amount']:.2f}")

# ============================
# --- Run Experiment Generation ---
# ============================
all_data = []
for participant_id in range(1, participants + 1):
    df = generate_trials(participant_id, training_reps=training_reps, testing_reps=testing_reps)
    all_data.append(df)
    compute_feature_correlations(df)
        # --- Add randomized 8-image column --- #
    fixed_images = [
        "Resources/T_B_S.png", "Resources/T_Y_S.png", "Resources/T_B_C.png", "Resources/T_Y_C.png",
        "Resources/N_B_S.png", "Resources/N_Y_S.png", "Resources/N_B_C.png", "Resources/N_Y_C.png"
    ]

    # Randomize the order once per participant
    rand_images = np.random.permutation(fixed_images).tolist()
    
    # Add the same randomized list to every row
    df["images_list"] = ",".join(rand_images)


    if save_csv:
        # Save combined, training-only, and testing-only CSVs
        out_path = os.path.join(output_dir, f"subj{participant_id:03d}_trials.csv")
        df.to_csv(out_path, index=False)
        df[df['phase'] == 'training'].to_csv(os.path.join(output_dir, f"subj{participant_id:03d}_training.csv"), index=False)
        if testing == 1:
            df[df['phase'] == 'testing'].to_csv(os.path.join(output_dir, f"subj{participant_id:03d}_testing.csv"), index=False)

# Combine all participants’ data
if save_csv:
    all_data_df = pd.concat(all_data, ignore_index=True)
    all_data_df.to_csv(os.path.join(output_dir, "all_trials.csv"), index=False)

print("\nExample output:")
print(df[['phase', 'tail', 'shape', 'color', 'food_amount', 'category', 'trial_num']])
