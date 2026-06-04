import argparse
import os
import pandas as pd
from tqdm import tqdm
from utils.constants import DATASET_PATH, useless_columns, meta_columns

def load_dataframe(preprocess: bool = False, save: bool = False, use_saved: bool = False):
    """Extract all the dataframes of a single simulation

        preprocess (bool, optional): if set, the df will be preprocessed
        save (bool, optional): if set, the df will be saved in /dataset/dataset.csv
        use_saved (bool, optional): if set, the df loaded straight from DATASET_PATH/dataset.csv

    Returns:
        pd.DataFrame: merged dataframe of all the simulations
    """
    if not use_saved:       
        dataframes = []
        csv_files = []

        # Walk through the directories and collect CSV file paths
        for sched_folder in os.listdir(DATASET_PATH):
            sched_path = os.path.join(DATASET_PATH, sched_folder)
            for tr_folder in os.listdir(sched_path):
                tr_path = os.path.join(sched_path, tr_folder)
                if os.path.isdir(tr_path):
                    for exp_folder in os.listdir(tr_path):
                        exp_path = os.path.join(tr_path, exp_folder)
                        if os.path.isdir(exp_path):
                            for bs_folder in os.listdir(exp_path):
                                bs_path = os.path.join(exp_path, bs_folder)
                                if os.path.isdir(bs_path):
                                    for slices_bs_folder in os.listdir(bs_path):
                                        slices_bs_path = os.path.join(bs_path, slices_bs_folder)
                                        if os.path.isdir(slices_bs_path):
                                            for file in os.listdir(slices_bs_path):
                                                if file.endswith("_metrics.csv"):
                                                    csv_file_path = os.path.join(slices_bs_path, file)
                                                    csv_files.append(csv_file_path)


        # Use tqdm to track the progress as you process the CSV files
        with tqdm(
            total=len(csv_files), desc="Concatenating CSV files", unit="file"
        ) as pbar:
            for csv_file_path in csv_files:
                df = pd.read_csv(csv_file_path)
                dataframes.append(df)
                # Update the progress bar after each file is processed
                pbar.update(1)

        df = pd.concat(dataframes, ignore_index=True)
        
        
        # Reward computation
        df["reward"] = df.apply(lambda row: compute_reward(row["sum_requested_prbs"], row["sum_granted_prbs"]), axis=1)
        
        # Normalization between -0.5 and 0.5
        new_min = -0.5
        new_max = 0.5
        min_reward = df["reward"].min()
        max_reward = df["reward"].max()

        df["reward"] = df["reward"].apply(
            lambda r: (r - min_reward) * (new_max - new_min) / (max_reward - min_reward) + new_min
        )

        if preprocess:
            print("Preprocess dataset")
            df = preprocess_dataset(df)
            
        if save:
            print("Save dataset")
            df.to_csv(os.path.join('/datasets', 'dataset.csv'), index=False)
    else:
        df = pd.read_csv(os.path.join('/datasets', 'dataset.csv'))

    return df


# Define a function to compute rewards based on conditions
def compute_reward(requested_prbs, granted_prbs):

    # requested_prbs == 0 and granted_prbs == 0: Encourage not giving PRBs when not needed
    if requested_prbs == 0 and granted_prbs == 0:
        return 0
    
    # requested_prbs == 0 and granted_prbs > 0: Penalize giving PRBs when not needed at all
    elif requested_prbs == 0 and granted_prbs > 0:
        return -1
    
    # requested_prbs < granted_prbs: Boundary condition, prevent over-granting (maybe we can add a penalty if that happens often)
    elif requested_prbs < granted_prbs:
        return 0
    
    # requested_prbs > 0: Calculate reward proportionally
    elif requested_prbs > 0:
        return -((requested_prbs - granted_prbs) / requested_prbs)   
    
    # Raise an error for any unhandled cases
    else:
        raise ValueError(f"Unhandled case: requested_prbs={requested_prbs}, granted_prbs={granted_prbs}")


def preprocess_dataset(df: pd.DataFrame):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Drop useless columns
    df = df.drop(useless_columns + meta_columns, axis=1)
    return df


if __name__ == '__main__':
    #######################
    # Parse data #
    #######################
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    df = load_dataframe(preprocess=True, save=True)