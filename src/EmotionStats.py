''' 
-------------------- Emotional alignment - Statistical Analysis --------------------

This script performs the statistical and descriptive analyses for the emotional
classification outputs (emotion_analysis.csv).

Specifically, the script:
1. Loads the emotion distributions and emotional similarity scores produced in the
   emotion classification pipeline.
2. Extracts the dominant ("top") emotion for each text type (Context, Human,
   GPT, and fine-tuned responses).
3. Extracts emotional similarity scores, and performs paired Wilcoxon signed-rank tests to compare emotional alignment
   between response types (Human vs GPT, Human vs FT, FT vs GPT).
4. Computes normalized emotion distributions across text types and visualizes
   them in a bar plot.

Input:
- emotion_analysis.csv  (output from the emotion classification script)

Output:
- Printed Wilcoxon test statistics
- emotion_distribution.png (bar plot of dominant emotion proportions)
'''

''' ----------------------- SETUP ----------------------'''
# Load packages
import pandas as pd
import ast
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# Load file
EMOTION_PATH = "/work/NLP2025/output/emotion_analysis.csv"
df = pd.read_csv(EMOTION_PATH)

# columns to be used for this part of the analysis
emotion_cols = ["Context_emotion", "Human_response_emotion", 
                "FT_response_emotion", "GPT_response_emotion"]

# convert the stored string-form dictionaries back intro Python dictionary objects
for col in emotion_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

''' ----------- Defining functions ----------- '''
# Defining function to identify the dominant (highest probability) emotion in each dictionary
def top_emotion(score_dict):
    if not isinstance(score_dict, dict) or len(score_dict) == 0:
        return None
    return max(score_dict, key=score_dict.get)

# append it to the dataframe
for prefix in ["Context", "Human_response", "FT_response", "GPT_response"]:
    df[f"{prefix}_top"] = df[f"{prefix}_emotion"].apply(top_emotion)


''' ---------------------- Statistics ------------------------'''
# extract the emotional similarity columns
# Higher value = the response is emotionally closer to the client’s emotion vector
human = df["Human_response_emotion_similarity"]
ft = df["FT_response_emotion_similarity"]
gpt = df["GPT_response_emotion_similarity"]

# Perform paired Wilcoxon signed-rank tests to compare emotional similarity
# across response types. Wilcoxon is appropriate because:
# - Similarity scores are paired (each response linked to a shared context)
# - Scores are bounded between 0 and 1
# - The distribution is non-normal

# Human vs GPT
w_human_gpt = wilcoxon(human, gpt)
# Human vs FT
w_human_ft = wilcoxon(human, ft)
# FT vs GPT
w_ft_gpt = wilcoxon(ft, gpt)

# prints the results
print("Human vs GPT:", w_human_gpt)
print("Human vs FT:", w_human_ft)
print("FT vs GPT:", w_ft_gpt)


''' -------------------- Emotion distribution comparison --------------------- '''
# Compute normalized emotion distributions across text types (proportions)
top_columns = ["Context_top", "Human_response_top", "FT_response_top", "GPT_response_top"]
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

distributions = {}

for col in top_columns:
    dist = df[col].value_counts(normalize=True).round(3)
    # reindex ensures missing emotions appear as 0
    distributions[col] = dist.reindex(emotion_labels, fill_value=0)
    

''' --------------- For plotting ----------------- '''

# Convert to dataframe
plot_df = pd.DataFrame(distributions, index=emotion_labels)

# Custom color palette
colors = {
    "Context_top": "#f05948",        # red
    "Human_response_top": "#5396c5",  # blue
    "FT_response_top": "#ff9c47",     # orange
    "GPT_response_top": "#69bc69"    # green
}

# Plot
plt.figure(figsize=(12, 6))
plot_df.plot(
    kind="bar",
    figsize=(12, 6),
    color=[colors[col] for col in top_columns] # apply the colors 
)

plt.title("Emotion Distribution Across Context and Responses", fontsize=16)
plt.xlabel("Emotion", fontsize=14)
plt.ylabel("Proportion", fontsize=14)
plt.legend(
    ["Context", "Human", "Fine-tuned (FT)", "GPT"],
    title="Source",
    fontsize=12
)

plt.xticks(rotation=45)
plt.tight_layout()

# save the plot in output path
output_path = "/work/NLP2025/output/emotion_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")