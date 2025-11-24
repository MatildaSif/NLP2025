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
from tqdm import tqdm

# Import custom utility functions
from utils import create_csv, load_data


''' Setup'''

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
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced", torch_dtype="auto")
    #model.to("cpu")
    return model


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
    for idx, context in tqdm(enumerate(df["Context"]), total=len(df),
                             desc="Generating responses", unit="row"):
        inputs = tokenizer(context, return_tensors="pt").to(model.device)
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

        # Print current ID safely
        current_id = df["ID"].iloc[idx]
        tqdm.write(f"Generated response for ID: {current_id}")

        #save responses to df in new column called "Response"
    df["FT_response"] = responses
    return df


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
    df = df[["Context", "ID"]]
    df = generate_responses(df, model, tokenizer)
    csv = create_csv(df, new_data_file, files_path)
    print("Responses saved to responses.csv")