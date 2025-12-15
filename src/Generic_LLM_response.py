# --------- API - GENERIC LLM RESPONSE GENERATION ------------
"""
This script generates GPT-based responses for a set of textual contexts stored
in a CSV-file. Each context is sent individually to the OpenAI API, and the
resulting model-generated responses are saved to a new CSV-file for later
analysis and comparison with human and fine-tuned model outputs.

What the script does
1. Loads an OpenAI API key from a local text-file (api_key.txt).
2. Performs a sanity check to confirm API is working.
3. Lists available OpenAI models.
4. Reads context data (ID + Context) from a CSV-file.
5. Builds standalone prompts for each row.
6. Sends each prompt to the specified GPT model.
7. Stores the generated responses in a new output CSV-file.

Input: context_ID.csv 
Output: GPT_responses.csv

"""


# --------- Setup ----------
#load packages
#from dotenv import load_dotenv
import os
import csv
from openai import OpenAI
from tqdm import tqdm

# Load API key - MAKE SURE TO HAVE A TXT-FILE CALLED api_key.txt IN NLP2025 IN ORDER TO RUN THIS
def load_api_key():
    try:
        with open("api_key.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError("API key file 'api_key.txt' not found. Please create it and put your API key inside.")

API_KEY = load_api_key()

client = OpenAI(api_key=API_KEY)

# ------------ TEST and model availability list------------
# sanity check to figure out whether API key works
response = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "Say hello"}]
)
print(response.choices[0].message.content)

# checking which models are available
models = client.models.list()
for m in models.data:
    print(m.id)

# --------------- GENERATING LLM RESPONSES ----------------
# define function to take prompt to LLM and return response
def call_gpt(prompt):
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",     # Input model heree
        messages=[{"role": "user", "content": prompt}], #states the prompt comes from the user and what the prompt is 
        max_tokens=200 # set max number of tokens here
    )
    return completion.choices[0].message.content #returns the first response for each prompt

# define function to build the prompt for each row while ignoring all previous contexts
def build_prompt(row):
    return f"""Ignore all previous messages.
Provide a standalone answer to this context and nothing else.

ID: {row['ID']}
Context: {row['Context']}
"""
# ------------- Running the functions ----------------
# opens the CSV-file
with open("/work/NLP2025/data/context_ID.csv", newline="", encoding="utf-8") as f:
    reader = list(csv.DictReader(f)) # read the file as a dictionary in a list to enable progression bars
    results = [] #creates empty list for results
    # loop to go through each row, build the prompt, create response from GPT, save the output
    for row in tqdm(reader, desc="Generating GPT responses"):
        prompt = build_prompt(row)
        response = call_gpt(prompt)
        results.append({"ID": row["ID"], "Context": row["Context"], "GPT_response": response})

# open/create output file 
with open("GPT_responses.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ID", "Context", "GPT_response"]) # make sure that the csv-file uses dictionaries as well
    writer.writeheader() # makes header row
    writer.writerows(results) #inputs results 