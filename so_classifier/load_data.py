import pandas as pd


def load_datas(p):
    df = pd.read_csv(p, sep='\t')
    return df

