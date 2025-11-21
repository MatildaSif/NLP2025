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

# Import custom utility functions
from utils import create_csv, load_data, get_model, get_tokenizer


''' Functions '''
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


def create_embeddings(encoder, df, text_columns = None):
    """
    Generate embeddings for each text column in the dataframe and store them
    in new columns named 'emb-<column>'.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and a variety of 'Response' columns.
        encoder(sentence_transformers.SentenceTransformer): A configured Hugging Face Sentence Transformer encoder
        text_columns (list, optional): List of column names to embed.
                                       If None, detects object/string columns.

    Returns:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and a variety of 'Response' columns, as well as an "Emb_----" for each of the text columns.
    """

    # If user didn’t specify, detect text columns automatically
    if text_columns is None:
        text_columns = [col for col in df.columns if df[col].dtype == "object"]

    # create embeddings and add to new columns
    for col in text_columns:
        print(f"creating embeddings for column {col}")

        df[f"emb-{col.lower()}"] = encoder.encode( 
            df[col].fillna("").tolist(),
            show_progress_bar = True
        )
        
    return df


''' Define Parameters '''
model_name = "all-MiniLM-L6-v2"
text_columns = ["Context", "Human_Response", "GPT_Response", "HF_Response"]
files_path = "data/"
data_file = "final.csv"


''' Main '''
if __name__ == "__main__":
    model_name = model_name
    text_columns = text_columns
    files_path = files_path
    data_file = data_file

    encoder = get_encoder(model_name)
    df = load_data(files_path, data_file) # from utils
   #df = df[["Context", "ID", ]] might not need or need to define the important columns
    df = create_embeddings(encoder, df, text_columns = text_columns)