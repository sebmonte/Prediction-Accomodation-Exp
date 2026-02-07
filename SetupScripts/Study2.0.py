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
participants = 220
save_csv = True
mac = False
#output_dir = r'/Users/sm6511/Desktop/Prediction-Accomodation-Exp/TrialFiles/Main2-7'
output_dir = r'C:\Users\Seb\Desktop\P-A Scripts\Prediction-Accomodation-Exp\TrialFiles\Main2-6-30'
os.makedirs(output_dir, exist_ok=True)
training_reps = 3
testing = 0 #Turn off testing for now (may not need it)
visualize = 0





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
wings = ["T", "N"]   # Wings / No Wings
colors = ["B", "Y"]  # Blue / Yellow 
tails = ["S", "C"]  # Straight / Curly

stimuli = []
for t, c, s in itertools.product(wings, colors, tails):
    stimuli.append({
        "wing": t,
        "color": c,
        "tail": s,
        "image_file": f"{t}_{c}_{s}.png"
    })

stim_df = pd.DataFrame(stimuli)

print(stim_df)




FEATURE_OPTIONS = {
    "wing":  [("T", "N"), ("N", "T")],
    "color": [("B", "Y"), ("Y", "B")],
    "tail": [("S", "C"), ("C", "S")]
}
direction_counts = {
    "wing": Counter(),
    "tail": Counter(),
    "color": Counter()
}
# ============================
# --- Feature Mapping and Counterbalancing ---
# ============================
# Randomizing which features are relevant (counterbalanced)

def feature_mapping(participant_id):
    dims = ['wing', 'color', 'tail']

    if participant_id % 2 == 1:
        if (participant_id // 2) % 2 == 0:
            relevant_dims = ("tail", "wing")
        else:
            relevant_dims = ("wing", "color")
    else:
        relevant_dims = ("tail", "color")

    irrelevant_dim = [d for d in dims if d not in relevant_dims][0]

    mapping = {
        "relevant_dims": list(relevant_dims),
        "irrelevant_dim": irrelevant_dim,
        "assignments": {}
    }

    # Counterbalance directions
    for dim in relevant_dims:
        options = FEATURE_OPTIONS[dim]

        # Build a relevance-conditioned index
        if dim == "wing":
            # wing direction flips every time tail is relevant
            idx = (participant_id // 2) % 2
        else:
            # tail/color: flip based on how often they appear as relevant
            idx = participant_id % 2

        high, low = options[idx]
        mapping["assignments"][dim] = {"high": high, "low": low}

    return mapping



# ============================
# --- Sampling Function ---
# ============================
def sample_additive(wing, tail, color, featuremap):
    ###FUNCTION FOR SAMPLING THE AMOUNT OF FOOD NEEDED FOR A GIVEN TRIAL
    #wing, tail, color: the wing/tail/color of the sperk for this trial
    #featuremap: the mapping between those features and what dimensions are relevant for this participant
    total = 0
    draw_idx = 1  # to label Draw 1, Draw 2

    #print("\nTRIAL:", f"wing={wing}, tail={tail}, color={color}")
    #print("Relevant features:", featuremap["relevant_dims"])
    #print("Assignments:", featuremap["assignments"])

    for dim, val in zip(["wing", "tail", "color"], [wing, tail, color]):

        if dim in featuremap["relevant_dims"]:

            if val == featuremap["assignments"][dim]["high"]:
                contrib = np.random.normal(mean_high, std_dev)
                level = "HIGH"
            else:
                contrib = np.random.normal(mean_low, std_dev)
                level = "LOW"

            contrib_clipped = np.clip(contrib, 0, 10)
            total += contrib_clipped

            #print(
            #    f"  Draw {draw_idx}: {dim}={val} → {level} → "
            #    f"{contrib_clipped:.3f}"
            #)

            draw_idx += 1

    #print(f"  Total: {total:.3f} → Final Food = {int(np.clip(round(total), 1, 10))}")
    #print("-" * 50)

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

    for val, label in zip(["T", "N"] if ir_dim == "wing"
                           else ["B", "Y"] if ir_dim == "color"
                           else ["S", "C"], ir_vals):
        rel_map[(ir_dim, val)] = label

    return rel_map


def map_relative_features(row, rel_map):
    return pd.Series({
        "wing_rel":  rel_map[("wing", row["wing"])],
        "color_rel": rel_map[("color", row["color"])],
        "tail_rel": rel_map[("tail", row["tail"])]
    })


def parse_image_name(img_name):
    base = os.path.basename(img_name).replace(".png", "")
    t, c, s = base.split("_")
    return {"wing": t, "color": c, "tail": s}

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
    #Return mean difference
    low_val   = means.index[0]
    high_val  = means.index[-1]
    mean_diff = means.iloc[-1] - means.iloc[0]


    return {"high": high_val, "low": low_val}, mean_diff


# ============================
# --- Trial Generation ---
# ============================
def generate_trials(fmap, training_reps):
    """
    Create a dataframe with randomized training trials for one participant.
    """
    mean_diff = 0
    while mean_diff < 0.5:
        training_trials = []
        for i in range(training_reps):
            shuffled = stim_df.sample(frac=1).reset_index(drop=True)
            shuffled["phase"] = "training"
            shuffled["rep"] = i + 1
            shuffled["trial_num"] = range(1 + i * len(shuffled), 1 + (i + 1) * len(shuffled))

            # Food amounts
            shuffled["food_amount"] = [
                sample_additive(t, s, c, fmap)
                for t, s, c in zip(shuffled["wing"], shuffled["tail"], shuffled["color"])
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
        ir_info, mean_diff = infer_irrelevant_high_low(training_df, fmap)
    
    training_df[f"{ir_dim}_high"] = ir_info["high"]
    training_df[f"{ir_dim}_low"]  = ir_info["low"]
    training_df["participant_id"] = participant_id
    training_df["food_image_file"] = training_df["food_amount"].astype(str) + "_food.png"

    return training_df


# ============================
# --- Run Experiment Generation ---
# ============================


while True:
        # --- Summary counters ---
    irrelevant_counts = Counter()  # how often each feature is irrelevant
    relevant_dir_counts = defaultdict(lambda: Counter())
    irrelevant_dir_counts = defaultdict(lambda: Counter())
    all_data = []
    irrelevant_dir_counts['wing'] = Counter()

    for participant_id in range(1, participants + 1):
        redo = True
        fmap = feature_mapping(participant_id)
        rel_map = build_relative_label_map(fmap)
        df = generate_trials(
                fmap,
                training_reps=training_reps
            )
            # ---- Track irrelevant feature ----
        irrelevant_counts[fmap["irrelevant_dim"]] += 1
        # ---- Track relevant feature directions ----
        for dim in fmap["relevant_dims"]:
            high = fmap["assignments"][dim]["high"]
            low  = fmap["assignments"][dim]["low"]

            relevant_dir_counts[dim][f"{high}_high"] += 1
            relevant_dir_counts[dim][f"{low}_low"]  += 1
        dim = fmap["irrelevant_dim"]
        high = df[f"{dim}_high"].iloc[0]
        low  = df[f"{dim}_low"].iloc[0]

        irrelevant_dir_counts[dim][f"{high}_high"] += 1
        irrelevant_dir_counts[dim][f"{low}_low"]   += 1
        wing_counts = irrelevant_dir_counts["wing"]
        all_data.append(df)
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
    balanced = False
    for wing in ["T", "N"]:
        high_key = f"{wing}_high"
        low_key  = f"{wing}_low"
        print(wing_counts[high_key])
        print(wing_counts[low_key])
        if wing_counts[high_key] == wing_counts[low_key]:
            #print(wing_counts[high_key])
            #print(wing_counts[low_key])
            balanced = True
    if balanced:
        break

    #print("\nPreview:")
    #print(df[['phase', 'wing', 'tail', 'color', 'food_amount', 'category', 'trial_num']])





print("\n================ SUMMARY ================\n")

print("Times each feature was IRRELEVANT:")
for dim in ["wing", "color", "tail"]:
    print(f"  {dim}: {irrelevant_counts[dim]}")

print("\nRelevant feature direction counts:")
for dim, counts in relevant_dir_counts.items():
    print(f"\n{dim.upper()}:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

print("\nirrelevant feature direction counts:")
for dim, counts in irrelevant_dir_counts.items():
    print(f"\n{dim.upper()}:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

total = participants

print("\n===== RELEVANCE BALANCE CHECK =====\n")

print("Wing relevant:", total - irrelevant_counts["wing"])
print("Wing irrelevant:", irrelevant_counts["wing"])
print("Proportion relevant:", 
      (total - irrelevant_counts["wing"]) / total)





ir_diff_rows = []

for df in all_data:
    participant = df["participant_id"].iloc[0]
    ir_dim = df["irrelevant_dim"].iloc[0]

    high_val = df[f"{ir_dim}_high"].iloc[0]
    low_val  = df[f"{ir_dim}_low"].iloc[0]

    df_high = df[df[ir_dim] == high_val]
    df_low  = df[df[ir_dim] == low_val]

    # Sanity check
    assert len(df_high) == len(df_low) == 12, \
        f"Participant {participant}: unexpected trial count"

    mean_high = df_high["food_amount"].mean()
    mean_low  = df_low["food_amount"].mean()
    diff = mean_high - mean_low

    ir_diff_rows.append({
        "participant_id": participant,
        "irrelevant_dim": ir_dim,
        "mean_high": mean_high,
        "mean_low": mean_low,
        "high_minus_low": diff
    })

ir_diff_df = pd.DataFrame(ir_diff_rows)

print("\n===== Irrelevant Feature High–Low Difference =====\n")

mean_diff = ir_diff_df["high_minus_low"].mean()
min_diff  = ir_diff_df["high_minus_low"].min()
max_diff  = ir_diff_df["high_minus_low"].max()
std_diff  = ir_diff_df["high_minus_low"].std()

print(f"Mean difference (high − low): {mean_diff:.3f}")
print(f"STD: {std_diff:.3f}")
print(f"Range: [{min_diff:.3f}, {max_diff:.3f}]")

rel_diff_rows = []
for df in all_data:
    participant = df["participant_id"].iloc[0]
    fmap = {
        "relevant_dims": [
            df["relevant_dim_1"].iloc[0],
            df["relevant_dim_2"].iloc[0]
        ]
    }

    for dim in fmap["relevant_dims"]:
        high_val = df[f"{dim}_high"].iloc[0]
        low_val  = df[f"{dim}_low"].iloc[0]

        df_high = df[df[dim] == high_val]
        df_low  = df[df[dim] == low_val]

        mean_high = df_high["food_amount"].mean()
        mean_low  = df_low["food_amount"].mean()
        diff = mean_high - mean_low

        rel_diff_rows.append({
            "participant_id": participant,
            "dimension": dim,
            "mean_high": mean_high,
            "mean_low": mean_low,
            "high_minus_low": diff
        })


rel_diff_df = pd.DataFrame(rel_diff_rows)



print("\n===== Relevant Feature High–Low Difference =====\n")

mean_diff = rel_diff_df["high_minus_low"].mean()
min_diff  = rel_diff_df["high_minus_low"].min()
max_diff  = rel_diff_df["high_minus_low"].max()
std_diff  = rel_diff_df["high_minus_low"].std()

print(f"Mean difference (high − low): {mean_diff:.3f}")
print(f"STD: {std_diff:.3f}")
print(f"Range: [{min_diff:.3f}, {max_diff:.3f}]")


plt.figure(figsize=(6, 4))

plt.hist(rel_diff_df["high_minus_low"],
         bins=20, alpha=0.6, label="Relevant", density=True)

plt.hist(ir_diff_df["high_minus_low"],
         bins=20, alpha=0.6, label="Irrelevant", density=True)

plt.axvline(0, linestyle="--")

plt.xlabel("High − Low")
plt.ylabel("Density")
plt.legend()
plt.title("Distribution of High − Low Differences")

plt.tight_layout()
plt.show()
