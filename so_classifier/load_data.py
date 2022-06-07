import pandas as pd


def load_data(p):
    df = pd.read_csv(p, sep='\t')
    return df

