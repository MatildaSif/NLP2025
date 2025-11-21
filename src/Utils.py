"""
Embeddings and data cleaning functions

This script contains functions for:
- Loading a tokeniser
- Loading a Hugging Face model
- Loading a data frame
- Creating a CSV from a dataframe

Dependencies:
- Pandas for data manipulation.
- Joblib for saving/loading models.
- Matplotlib for confusion matrix visualization.
"""

# ------------------------ Imports ------------------------
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import transformers
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


# ------------------------ Functions ------------------------

def load_data(files_path, data_file):
    """
    Load and preprocess a dataset of human prompts from a CSV file.

    Parameters:
        files_path (str): The directory where the file is stored.
        data_file (str): Filename of the CSV containing labeled comments.

    Returns:
        df (pd.DataFrame): DataFrame with 'Context' and 'ID' columns.
    """
    data_path = os.path.join(files_path, data_file)
    df = pd.read_csv(data_path, quotechar='"')
    print(f"Original data size: {len(df)}")
    return df


def create_csv(df, new_data_file):
    """
    Create csv from new data frame and save to data file path.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and 'Response' columns.
        new_data_file (str): Name of new csv file

    Returns:
        csv_data (csv): New saved csv file
    """
    df__new = df.copy()
    csv_data = df_new.to_csv(new_data_file, index = False)
    return csv_data