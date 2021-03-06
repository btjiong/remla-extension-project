import numpy as np
import pandas as pd


def load_data(p):
    """
    Loads the data from a directory as a DataFrame
    """
    df = pd.read_csv(p, sep="\t")
    return df


def save_data(p, df):
    """
    Saves the data as a .tsv file in a directory
    """
    print("Now saving")
    df.to_csv(p, index=False, sep="\t")
    print("saved checked .tsv test file\n")


def split_data(df):
    """
    Shuffles and splits (67/20/13) the given DataFrame
    """
    # Shuffles the data
    df = df.sample(frac=1, random_state=1)

    # Splits the data 67/20/13
    train, validate, test = np.split(df, [int(0.67 * len(df)), int(0.87 * len(df))])
    return train, validate, test


def concat_data(df1, df2):
    """
    Concatenates two DataFrames and resets the index
    """
    # Concat the two datasets
    df = pd.concat([df1, df2])

    # Resets the index
    df.reset_index(drop=True, inplace=True)

    print(f"Concatenated file: {df.shape}")

    return df
