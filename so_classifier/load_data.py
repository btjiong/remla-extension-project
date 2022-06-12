import pandas as pd


def load_data(p):
    df = pd.read_csv(p, sep="\t")
    return df


# saving the validated dataframe as .tsv
def save_data(p, df):
    print("Now saving")
    df.to_csv(p, index=False, sep="\t")
    print("saved checked .tsv test file\n")
