"""
Document Embeddings

This script applies a BERT model sentence transformer that creates text embeddings.

Goals:
- Generate embeddings for all columns and rows in dfs.
- A new df with embeddings for all contexts and responses

"""

''' Setup '''
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import sentence_transformers
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from functools import reduce
import numpy as np


# Import custom utility functions
from utils import load_data, create_csv


''' Functions '''
def merge_dfs(dfs, files_path):
    """
    Merge multiple DataFrames on 'Context'.

    Parameters:
        dfs (list of pd.DataFrame): List of DataFrames to merge. Each must have 'Context'

    Returns:
        pd.DataFrame: Merged DataFrame with renamed response columns called "final_df.csv"
    """
    if not dfs:
        raise ValueError("The list of DataFrames is empty")
    
    final_df = reduce(lambda left, right: pd.merge(left, right, on=["Context", "ID"], how="inner"), dfs)

    final_df = final_df[["ID","Context","Human_response", "FT_response", "GPT_response"]]
    # Make ID first column
    cols = final_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("ID")))
    final_df = final_df[cols]

    create_csv(final_df, "final_df.csv", files_path)
    return final_df


def get_encoder(model_name):
    """
    Get Encoder

    Parameters:
        model_name (str): The name or path of the pretrained model.

    Returns:
        encoder (sentence_transformers.SentenceTransformer): A configured Hugging Face Sentence Transformer encoder
    """
    encoder = SentenceTransformer(model_name)
    return encoder


def create_embeddings(encoder, df, text_columns = None, save_path = None):
    """
    Generate embeddings for each text column in the dataframe and store them
    in new columns named 'emb-<column>'.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and a variety of 'Response' columns.
        encoder(sentence_transformers.SentenceTransformer): A configured Hugging Face Sentence Transformer encoder
        text_columns (list, optional): List of column names to embed.
                                       If None, detects object/string columns.
        save_path (str): folder to save .npy files

    Returns:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and a variety of 'Response' columns, as well as an "Emb_----" for each of the text columns.
    """

    # If user didn’t specify, detect text columns automatically
    if text_columns is None:
        text_columns = [col for col in df.columns if df[col].dtype == "object"]

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # create embeddings and add to new columns
    for col in text_columns:
        print(f"creating embeddings for column {col}")
        
        # Compute embeddings (shape: n_rows x 384)
        emb_matrix = encoder.encode(
            df[col].fillna("").tolist(),
            show_progress_bar=True
        )
        # Save as new column in DataFrame as lists (optional)
        df[f"emb-{col.lower()}"] = emb_matrix.tolist()

        # Save embeddings as NumPy array
        np.save(os.path.join(save_path, f"emb-{col.lower()}.npy"), emb_matrix)

    return df




''' Define Parameters '''
model_name = "all-MiniLM-L6-v2"
text_columns = ["Context", "Human_response", "GPT_response", "FT_response"]
files_path = "data/"
data_file_list = ["context_ID.csv", "Human_responses.csv", "GPT_responses.csv", "FT_Responses.csv"]
data_file = "final_df.csv"
save_path = "data/emb/"


''' Main '''
if __name__ == "__main__":
    model_name = model_name
    text_columns = text_columns
    files_path = files_path
    data_file = data_file
    save_path = save_path

    dfs = []
    for file in data_file_list:
        df = load_data(files_path, file)
        dfs.append(df)

    df = merge_dfs(dfs, files_path) # creates df called final_df.csv
    encoder = get_encoder(model_name)
    df = load_data(files_path, data_file) # from utils
    df = create_embeddings(encoder, df, text_columns = text_columns, save_path = save_path)