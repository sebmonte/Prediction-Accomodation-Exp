# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 09:48:43 2025

@author: Seb
"""

import itertools
import numpy as np
import pandas as pd
import os

# parameters for food distribution
mean_high = 7.5
mean_low = 3.5
std_dev = 2  # Spread
participants = 2
save_csv = False
output_dir = r'C:\Users\Seb\Desktop\TestExp\Trial_Files'
os.makedirs(output_dir, exist_ok=True)
training_reps = 2
testing_reps = 1

def sample(distribution="high"):
    '''Return one integer sample from the 'High' or 'Low' Food Distribution'''
    mean = mean_high if distribution == "high" else mean_low
    x = np.random.normal(mean, std_dev)
    return int(np.clip(round(x), 1, 10))

# Coding for features 
tails = ["T", "N"]   # Tail / No Tail
colors = ["B", "Y"]  # Red / Blue
shapes = ["S", "C"]  # Square / Circle

stimuli = []
for t, c, s in itertools.product(tails, colors, shapes):
    category = "high" if (t == "T" and s == "S") else "low"
    stimuli.append({
        "tail": t,
        "color": c,
        "shape": s,
        "image_file": f"{t}_{c}_{s}.png",
        "category": category
    })

stim_df = pd.DataFrame(stimuli)

# --- generate trials ---
def generate_trials(participant_id, training_reps=2, testing_reps=2):
    """
    Create a dataframe with randomized training and testing trials for one participant.
    """
    training_trials = []
    for i in range(training_reps):
        shuffled = stim_df.sample(frac=1).reset_index(drop=True)
        shuffled["phase"] = "training"
        shuffled["rep"] = i + 1
        shuffled["trial_num"] = range(1 + i * len(shuffled), 1 + (i + 1) * len(shuffled))
        shuffled["food_amount"] = [sample(cat) for cat in shuffled["category"]]
        training_trials.append(shuffled)

    training_df = pd.concat(training_trials, ignore_index=True)

    testing_trials = []
    start_trial = len(training_df)
    for i in range(testing_reps):
        shuffled_test = stim_df.sample(frac=1).reset_index(drop=True)
        shuffled_test["phase"] = "testing"
        shuffled_test["rep"] = i + 1
        shuffled_test["trial_num"] = range(start_trial + 1, start_trial + 1 + len(shuffled_test))
        shuffled_test["food_amount"] = [sample(cat) for cat in shuffled_test["category"]]
        start_trial += len(shuffled_test)
        testing_trials.append(shuffled_test)

    testing_df = pd.concat(testing_trials, ignore_index=True)

    all_trials = pd.concat([training_df, testing_df], ignore_index=True)
    all_trials["participant_id"] = participant_id  # ← add participant ID to every row
    all_trials["food_image_file"] = all_trials["food_amount"].astype(str) + "_food.png"
    return all_trials



def compute_feature_correlations(df):
    """Compute correlations of features and stimulus averages with food needed for training and testing phases."""
    
    participant = df['participant_id'].iloc[0]

    # --- Training phase ---
    df_train = df[df['phase'] == 'training'].copy()
    df_train["tail_num"] = df_train["tail"].map({"T": 1, "N": 0})
    df_train["color_num"] = df_train["color"].map({"B": 1, "Y": 0})
    df_train["shape_num"] = df_train["shape"].map({"S": 1, "C": 0})
    
    # Feature correlations
    corr_tail_train = df_train["tail_num"].corr(df_train["food_amount"])
    corr_color_train = df_train["color_num"].corr(df_train["food_amount"])
    corr_shape_train = df_train["shape_num"].corr(df_train["food_amount"])
    
    print(f"\nTraining phase correlations for participant {participant}:")
    print(f"  Tail:  {corr_tail_train:.3f}")
    print(f"  Color: {corr_color_train:.3f}")
    print(f"  Shape: {corr_shape_train:.3f}")
    
    # --- Stimulus-level averages ---
    stim_means_train = (
        df_train.groupby("image_file", as_index=False)["food_amount"]
        .mean()
        .rename(columns={"food_amount": "mean_food_amount"})
    )

    print("\nAverage food amount per stimulus (training):")
    for _, row in stim_means_train.iterrows():
        print(f"  {row['image_file']}: {row['mean_food_amount']:.2f}")

    # --- Testing phase ---
    df_test = df[df['phase'] == 'testing'].copy()
    df_test["tail_num"] = df_test["tail"].map({"T": 1, "N": 0})
    df_test["color_num"] = df_test["color"].map({"B": 1, "Y": 0})
    df_test["shape_num"] = df_test["shape"].map({"S": 1, "C": 0})
    
    corr_tail_test = df_test["tail_num"].corr(df_test["food_amount"])
    corr_color_test = df_test["color_num"].corr(df_test["food_amount"])
    corr_shape_test = df_test["shape_num"].corr(df_test["food_amount"])
    
    print(f"\nTesting phase correlations for participant {participant}:")
    print(f"  Tail:  {corr_tail_test:.3f}")
    print(f"  Color: {corr_color_test:.3f}")
    print(f"  Shape: {corr_shape_test:.3f}")
    
    stim_means_test = (
        df_test.groupby("image_file", as_index=False)["food_amount"]
        .mean()
        .rename(columns={"food_amount": "mean_food_amount"})
    )

    print(f"\nAverage food amount per stimulus (testing) participant: {participant_id}:")
    for _, row in stim_means_test.iterrows():
        print(f"  {row['image_file']}: {row['mean_food_amount']:.2f}")

    return (
        (corr_tail_train, corr_color_train, corr_shape_train, stim_means_train),
        (corr_tail_test, corr_color_test, corr_shape_test, stim_means_test)
    )


all_data = []
for participant_id in range(1, participants + 1):
    df = generate_trials(participant_id, training_reps=training_reps, testing_reps=testing_reps)
    all_data.append(df)

    compute_feature_correlations(df)

    if save_csv:
        # save combined trials
        out_path = os.path.join(output_dir, f"subj{participant_id:03d}_trials.csv")
        df.to_csv(out_path, index=False)
    
        # --- save training-only CSV ---
        train_path = os.path.join(output_dir, f"subj{participant_id:03d}_training.csv")
        df[df['phase'] == 'training'].to_csv(train_path, index=False)
    
        # --- save testing-only CSV ---
        test_path = os.path.join(output_dir, f"subj{participant_id:03d}_testing.csv")
        df[df['phase'] == 'testing'].to_csv(test_path, index=False)


if save_csv:
    # concatenate all participants' data into a single DataFrame
    all_data_df = pd.concat(all_data, ignore_index=True)
    
    # save it out
    out_path_all = os.path.join(output_dir, "all_trials.csv")
    all_data_df.to_csv(out_path_all, index=False)


print(df[['phase', 'food_amount', 'trial_num']])
