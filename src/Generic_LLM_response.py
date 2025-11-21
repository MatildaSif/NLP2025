# --------- API - GENERIC LLM RESPONSE GENERATION ------------

# --------- Setup ----------
#load packages
from dotenv import load_dotenv
import os
import csv
from openai import OpenAI

#note that you have to have a .env file with the API key for producing generic LLM responses
load_dotenv()  # reads .env file automatically 

# Retrieves the API key and creates the client that we will use to send prompts to GPT models
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
Answer only this context and nothing else.

ID: {row['ID']}
Context: {row['Context']}
"""
# ------------- Running the functions ----------------
# opens the CSV-file
with open("context_ID.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f) #reads the file as a dictionary
    results = [] #creates empty list for results
    #loop to go through each row, build the prompt, create response from GPT, save the output
    for row in reader:
        prompt = build_prompt(row)
        response = call_gpt(prompt)
        results.append({"ID": row["ID"], "Context": row["Context"], "GPT_response": response})

# open/create output file 
with open("GPT_responses.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["ID", "Context", "GPT_response"]) # make sure that the csv-file uses dictionaries as well
    writer.writeheader() # makes header row
    writer.writerows(results) #inputs results 