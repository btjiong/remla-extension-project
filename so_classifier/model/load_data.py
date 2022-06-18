import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


def load_data(p):
    df = pd.read_csv(p, sep="\t")
    return df


# saving the validated dataframe as .tsv
def save_data(p, df):
    print("Now saving")
    df.to_csv(p, index=False, sep="\t")
    print("saved checked .tsv test file\n")


def split_data(df):
    # Shuffles the data
    df = df.sample(frac=1, random_state=1)

    # Splits the data 67/20/13
    train, validate, test = np.split(df, [int(0.67 * len(df)), int(0.87 * len(df))])
    return train, validate, test


def concat_data(df1, df2):
    # Concat the two datasets
    df = pd.concat([df1, df2])

    # Resets the index
    df.reset_index(drop=True, inplace=True)

    print(f"Concatenated file: {df.shape}")

    return df


# Set up Google Drive API
SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "so_classifier/credentials.json"

credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
service = build("drive", "v3", credentials=credentials)

file_id = "1Ds8RLIU4lXzV-HYqoqGgKTEebfMJC_Bq"

file_metadata = {
    "name": "tfidf_vectorizer.joblib",
    "parents": ["1InFeBLhOU-Y2Sj8mjE-2IKFs9H5rby3K"]
}

media_body = MediaFileUpload("model/tfidf_vectorizer.joblib", mimetype="application/octet-stream")

service.files().update(
    fileId=file_id,
    media_body=media_body
).execute()
