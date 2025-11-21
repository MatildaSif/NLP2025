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

''' DF for prompting LLMS'''

# ---------- Load Data -----------
data = pd.read_csv("/data/convertcsv.csv")

# ---------- Data Cleaning ----------
# Remove 'Response' column if it exists
if "Response" in data.columns:
    data_cleaned = data.drop(columns=["Response"])
else:
    data_cleaned = data.copy()

# Keep only unique Context values and drop rows with missing Context
df = data_cleaned.dropna(subset=["Context"]).drop_duplicates(subset=["Context"])

# ---------- Add ID Column ----------
df["ID"] = ["context_" + str(i+1) for i in range(len(df))]

# Reorder columns to have ID first
df = df[["ID", "Context"]]

# ---------- Save Cleaned CSV ----------
df.to_csv("context_ID.csv", index=False)




''' Df with selected Human Response'''

# ---------- Load Data -----------
data = pd.read_csv("/data/convertcsv.csv")

