"""
Mental Health Chatbot Hugging Face Model Script

This script applies a HF Model that will generate text from real human prompts regarding mental health

Goals:
- Generate answers from a Hugging face model to real human mental health prompts
"""

''' Setup'''
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import transformers
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch


''' Functions '''
def get_tokenizer(model_name):
    """
    Get  Tokeniser

    Parameters:
        model_name (str): The name or path of the pretrained model.

    Returns:
        tokeniser (transformers.AutoTokeniser): A configured Hugging Face tokeniser
    """
    return AutoTokenizer.from_pretrained(model_name)

def get_model(model_name):
    """
    Get Hugging Face model

    Parameters:
        model_name (str): The name or path of the pretrained model.

    Returns:
        model (transformers.AutoModelForCausalLM): A configured Hugging Face model
    """
    return AutoModelForCausalLM.from_pretrained(model_name)



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
    df = df[["Context", "ID"]]
    print(f"Original data size: {len(df)}")
    return df


def generate_responses(df, model, tokenizer):
    """
    Generate responses with the HF model based on context's in rows in a dataset of human prompts from a CSV file.
    Save the responses to the same dataframe in a new column called "Response"

    Parameters:
        df (pd.DataFrame): DataFrame with 'Context' and 'ID' columns.
        model (transformers.AutoModelForCausalLM): A configured Hugging Face model
        tokenizer (transformers.AutoTokeniser): A configured Hugging Face tokenizer

    Returns:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and 'Response' columns.
    """
    responses = []
    for context in df["Context"]:
        inputs = tokenizer(context, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append(response)
        #save responses to df in new column called "Response"
    df["Response"] = response
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


''' Define Parameters'''
model_name = "tanusrich/Mental_Health_Chatbot"


'''Main'''
if __name__ == "__main__":
    model_name = model_name
    files_path = "data/"
    data_file = "context_ID.csv"
    new_data_file = "HF_Responses.csv"

    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    df = load_data(files_path, data_file)
    df = generate_responses(df, model, tokenizer)
    csv = create_csv(df, new_data_file)