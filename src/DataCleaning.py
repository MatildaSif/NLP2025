"""
DataCleaning.py

This script does 2 things:
1. It creates a csv that can be used for prompting the LLM models so the human responses are removed
    - Load CSV
    - Remove 'Response' column
    - Keep unique Context values
    - Drop missing values
    - Add ID column
    - Save cleaned CSV

2. It creates a csv with the human responses, where only one response is kept per context
    - Load CSV
    - Select the Response for each context with the most likes
    - Remove the other responses for each context
    - Save cleaned CSV

"""

import pandas as pd
from utils import load_data, create_csv

if __name__ == "__main__":

    ''' Df with selected Human Response'''

    # ---------- Load Data -----------
    data = pd.read_csv("data/Original_Human_responses.csv")

    #Select important columns
    data = data[["questionID","questionTitle", "questionText","answerText", "upvotes"]]

    # combine text and title to one column for prompting
    data["questionText"] = data["questionTitle"].fillna("").astype(str) + " " + data["questionText"].fillna("").astype(str)

    # for each question keep those with the max upvotes
    data["MaxUpvote"] = data.groupby("questionText")["upvotes"].transform("max")

    # select only responses with max upvotes
    top_responses = data[data["upvotes"] == data ["MaxUpvote"]].copy()

    # select one response if there are multiple with max upvote for that context
    final_df = top_responses.groupby("questionText").sample(n=1, random_state =42)

    # drop uneeded column
    final_df = final_df.drop(columns=["MaxUpvote", "upvotes", "questionTitle"]).reset_index(drop=True)

    # Rename columns
    final_df = final_df.rename(columns = {"questionID": "ID", "answerText": "Human_response", "questionText": "Context"})

    # rename ID column to match later dfs
    final_df["ID"] = ["context_" + str(i+1) for i in range(len(final_df))]

    create_csv(final_df, "Human_responses.csv", "data/")



    ''' DF for prompting LLMS'''
        
    # ---------- Load Data -----------
    data = pd.read_csv("data/Human_responses.csv")

    # ---------- Data Cleaning ----------
    # Remove 'Response' column if it exists
    if "Human_response" in data.columns:
        data_cleaned = data.drop(columns=["Human_response"])
    else:
        data_cleaned = data.copy()

    # Keep only unique Context values and drop rows with missing Context
    df = data_cleaned.dropna(subset=["Context"]).drop_duplicates(subset=["Context"])

    # Reorder columns to have ID first
    df = df[["ID", "Context"]]

    # ---------- Save Cleaned CSV ----------
    df.to_csv("data/context_ID.csv", index=False)

