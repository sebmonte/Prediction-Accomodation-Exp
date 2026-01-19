# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 2025

@author: Sebastian Montesinos

Produces the sequence of trials for each participant, using an additive rule:
- What features are assigned to be relevant are randomly assigned
"""

import itertools
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
import json
from collections import Counter, defaultdict

# ============================
# --- Parameters ---
# ============================
mean_high = 3.5   # Means of distributions to draw from
mean_low = 1.5
std_dev = 1   # Spread of each distribution
participants = 150
save_csv = False
output_dir = r'C:\Users\Seb\Desktop\TestExp\Trial_Files' #Output path
output_dir = r'C:\Users\Seb\Desktop\P-A Scripts\Prediction-Accomodation-Exp\TrialFiles\Pilot-1-09'
os.makedirs(output_dir, exist_ok=True)
training_reps = 3
testing = 0 #Turn off testing for now (may not need it)
visualize = 1


# --- Summary counters ---
irrelevant_counts = Counter()  # how often each feature is irrelevant
relevant_dir_counts = defaultdict(lambda: Counter())


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

if visualize:
    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.bar(values - 0.2, low_counts / n_samples, width=0.4, alpha=0.7, label="Low")
    plt.bar(values + 0.2, high_counts / n_samples, width=0.4, alpha=0.7, label="High")
    plt.xticks(values)
    plt.xlabel("Food value")
    plt.ylabel("Probability")
    plt.title("Selection Distributions")
    plt.legend()
    plt.tight_layout()
    plt.show()



# ============================
# --- Stimulus Setup ---
# ============================
tails = ["T", "N"]   # Curly Tail / Straight Tail
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

print(stim_df)

# ============================
# --- Feature Mapping and Counterbalancing ---
# ============================
# Randomizing which features are relevant

def feature_mapping():
    dims = ['tail', 'color', 'shape']
    relevant_dims = random.sample(dims, 2) #Randomly choose the two relevant dimensions
    irrelevant_dim = [d for d in dims if d not in relevant_dims][0] #The other dimension is the irrelevant one
    mapping = {"relevant_dims": relevant_dims, "irrelevant_dim": irrelevant_dim, "assignments": {}}
    options = {
        "tail": ["T", "N"],
        "color": ["B", "Y"],
        "shape": ["S", "C"]
    }
    #Randomly choose which of the dimensions of the feature is relevant
    for dim in relevant_dims:
        high_val = random.choice(options[dim]) #Pick the high one randomly
        low_val = [x for x in options[dim] if x != high_val][0] #The other becomes the low value
        mapping["assignments"][dim] = {"high": high_val, "low": low_val}

    return mapping

# ============================
# --- Sampling Function ---
# ============================
def sample_additive(tail, shape, color, featuremap):
    ###FUNCTION FOR SAMPLING THE AMOUNT OF FOOD NEEDED FOR A GIVEN TRIAL
    #tail, shape, color: the tail/shape/color of the sperk for this trial
    #featuremap: the mapping between those features and what dimensions are relevant for this participant
    total = 0
    draw_idx = 1  # to label Draw 1, Draw 2

    print("\nTRIAL:", f"tail={tail}, shape={shape}, color={color}")
    print("Relevant features:", featuremap["relevant_dims"])
    print("Assignments:", featuremap["assignments"])

    for dim, val in zip(["tail", "shape", "color"], [tail, shape, color]):

        if dim in featuremap["relevant_dims"]:

            if val == featuremap["assignments"][dim]["high"]:
                contrib = np.random.normal(mean_high, std_dev)
                level = "HIGH"
            else:
                contrib = np.random.normal(mean_low, std_dev)
                level = "LOW"

            contrib_clipped = np.clip(contrib, 0, 10)
            total += contrib_clipped

            print(
                f"  Draw {draw_idx}: {dim}={val} → {level} → "
                f"{contrib_clipped:.3f}"
            )

            draw_idx += 1

    print(f"  Total: {total:.3f} → Final Food = {int(np.clip(round(total), 1, 10))}")
    print("-" * 50)

    return int(np.clip(round(total), 1, 10))


# ============================
# --- Label Categories in dataframe ---
# ============================
def label_category(row, fmap):
    #3 possible categories for amount of food sperks need: low, medium or high. This function takes in a row and determines what category
    #A given sperk falls into
    score = 0
    for dim in fmap["relevant_dims"]:
        if row[dim] == fmap["assignments"][dim]["high"]:
            score += 1
    return ["low", "medium", "high"][score]

def build_relative_label_map(fmap):
    """
    Creates an arbitrary but fixed mapping from feature values
    to relative labels (H1/H2/L1/L2/I1/I2) for one participant.
    """
    rel_map = {}

    # Arbitrarily assign R1 / R2 to the two relevant dimensions
    rel_dims = fmap["relevant_dims"]
    random.shuffle(rel_dims)  # arbitrary but fixed

    for idx, dim in enumerate(rel_dims, start=1):
        high = fmap["assignments"][dim]["high"]
        low  = fmap["assignments"][dim]["low"]

        rel_map[(dim, high)] = f"H{idx}"
        rel_map[(dim, low)]  = f"L{idx}"

    # Irrelevant dimension
    ir_dim = fmap["irrelevant_dim"]
    ir_vals = ["I1", "I2"]
    random.shuffle(ir_vals)

    for val, label in zip(["T", "N"] if ir_dim == "tail"
                           else ["B", "Y"] if ir_dim == "color"
                           else ["S", "C"], ir_vals):
        rel_map[(ir_dim, val)] = label

    return rel_map

'''
def map_relative_features(row, fmap):
    out = {}
    labeling = random.randint(0, 1)
    for dim in ["tail", "color", "shape"]:
        if dim in fmap["relevant_dims"]:
            if row[dim] == fmap["assignments"][dim]["high"]:
                out[dim + "_rel"] = "R1"
            else:
                out[dim + "_rel"] = "L1"
        else:
            out[dim + "_rel"] = "IR"

    return pd.Series(out)
'''

def map_relative_features(row, rel_map):
    return pd.Series({
        "tail_rel":  rel_map[("tail", row["tail"])],
        "color_rel": rel_map[("color", row["color"])],
        "shape_rel": rel_map[("shape", row["shape"])]
    })


def parse_image_name(img_name):
    base = os.path.basename(img_name).replace(".png", "")
    t, c, s = base.split("_")
    return {"tail": t, "color": c, "shape": s}

def label_image_sequence(images_list_str, fmap):
    img_list = images_list_str.split(",")

    categories = []
    for img in img_list:
        row = parse_image_name(img)
        cat = label_category(row, fmap)
        categories.append(cat)

    return ",".join(categories)

def infer_irrelevant_high_low(df, fmap):
    """
    Infer high/low labels for the irrelevant dimension based on
    empirical mean food amount in training data.
    """
    ir_dim = fmap["irrelevant_dim"]

    means = (
        df.groupby(ir_dim)["food_amount"]
        .mean()
        .sort_values()
    )

    low_val  = means.index[0]
    high_val = means.index[-1]
    print(means)

    return {"high": high_val, "low": low_val}


# ============================
# --- Trial Generation ---
# ============================
def generate_trials(participant_id, training_reps):
    """
    Create a dataframe with randomized training trials for one participant.
    """
    training_trials = []
    fmap = feature_mapping()
    rel_map = build_relative_label_map(fmap)
    

    for i in range(training_reps):
        shuffled = stim_df.sample(frac=1).reset_index(drop=True)
        shuffled["phase"] = "training"
        shuffled["rep"] = i + 1
        shuffled["trial_num"] = range(1 + i * len(shuffled), 1 + (i + 1) * len(shuffled))

        # Food amounts
        shuffled["food_amount"] = [
            sample_additive(t, s, c, fmap)
            for t, s, c in zip(shuffled["tail"], shuffled["shape"], shuffled["color"])
        ]
        # Categories
        shuffled["category"] = [
            label_category(row, fmap) for _, row in shuffled.iterrows()
        ]

        # Save feature mapping
        shuffled["relevant_dim_1"] = fmap["relevant_dims"][0]
        shuffled["relevant_dim_2"] = fmap["relevant_dims"][1]
        shuffled["irrelevant_dim"] = fmap["irrelevant_dim"]

        for dim in fmap["relevant_dims"]:
            shuffled[f"{dim}_high"] = fmap["assignments"][dim]["high"]
            shuffled[f"{dim}_low"]  = fmap["assignments"][dim]["low"]
        relative_cols = shuffled.apply(lambda row: map_relative_features(row, rel_map), axis=1)
        shuffled = pd.concat([shuffled, relative_cols], axis=1)

        training_trials.append(shuffled)

    training_df = pd.concat(training_trials, ignore_index=True)
    ir_dim = fmap["irrelevant_dim"]
    ir_assignment = infer_irrelevant_high_low(training_df, fmap)
    training_df[f"{ir_dim}_high"] = ir_assignment["high"]
    training_df[f"{ir_dim}_low"]  = ir_assignment["low"]
    training_df["participant_id"] = participant_id
    training_df["food_image_file"] = training_df["food_amount"].astype(str) + "_food.png"

    return training_df, fmap


# ============================
# --- Run Experiment Generation ---
# ============================
all_data = []
for participant_id in range(1, participants + 1):
    df, fmap = generate_trials(participant_id, training_reps=training_reps)
        # ---- Track irrelevant feature ----
    irrelevant_counts[fmap["irrelevant_dim"]] += 1

    # ---- Track relevant feature directions ----
    for dim in fmap["relevant_dims"]:
        high = fmap["assignments"][dim]["high"]
        low  = fmap["assignments"][dim]["low"]

        relevant_dir_counts[dim][f"{high}_high"] += 1
        relevant_dir_counts[dim][f"{low}_low"]  += 1

    all_data.append(df)
    #correlations = correlations_training = compute_feature_correlations(df)
    #out_path2 = os.path.join(output_dir_pilot, f"subj{participant_id:03d}_trials.csv")
    #correlations.to_csv(out_path2, index = False)
        # Below adds the paths to all the images in a randomized order for testing
    fixed_images = [
        "Resources/T_B_S.png", "Resources/T_Y_S.png", "Resources/T_B_C.png", "Resources/T_Y_C.png",
        "Resources/N_B_S.png", "Resources/N_Y_S.png", "Resources/N_B_C.png", "Resources/N_Y_C.png"
    ]

    rand_images = np.random.permutation(fixed_images).tolist()
    df["images_list"] = ",".join(rand_images)
    df["testing_categories"] = df["images_list"].apply(lambda x: label_image_sequence(x, fmap))



    if save_csv:
        out_path = os.path.join(output_dir, f"subj{participant_id:03d}_training.csv")
        df.to_csv(out_path, index=False)

# Combine all participants’ data
if testing:
    all_data_df = pd.concat(all_data, ignore_index=True)
    all_data_df.to_csv(os.path.join(output_dir, "all_trials.csv"), index=False)

print("\nPreview:")
print(df[['phase', 'tail', 'shape', 'color', 'food_amount', 'category', 'trial_num']])





print("\n================ SUMMARY ================\n")

print("Times each feature was IRRELEVANT:")
for dim in ["tail", "color", "shape"]:
    print(f"  {dim}: {irrelevant_counts[dim]}")

print("\nRelevant feature direction counts:")
for dim, counts in relevant_dir_counts.items():
    print(f"\n{dim.upper()}:")
    for k, v in counts.items():
        print(f"  {k}: {v}")









'''

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
'''