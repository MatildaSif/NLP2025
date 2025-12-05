''' 
-------------------- Emotion Classification - Statistical Analysis -----------------------
Okay, so for this script we will be doing some statistics on the emotional Classification investigation.

The script will:
- 

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

# convert stringified dicts to real dicts
for col in emotion_cols:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

''' ----------- Definin functions ----------- '''
# Defining function to find the top emotion
def top_emotion(score_dict):
    if not isinstance(score_dict, dict) or len(score_dict) == 0:
        return None
    return max(score_dict, key=score_dict.get)

# append it to the dataframe
for prefix in ["Context", "Human_response", "FT_response", "GPT_response"]:
    df[f"{prefix}_top"] = df[f"{prefix}_emotion"].apply(top_emotion)

# Here, each line compares two columns row-by-row and creates a new boolean column where:
# TRUE = if, for that row, the respondents top emotion matches the clients top emotion
# False otherwise
df["Human_match"] = df["Context_top"] == df["Human_response_top"]
df["FT_match"] = df["Context_top"] == df["FT_response_top"]
df["GPT_match"] = df["Context_top"] == df["GPT_response_top"]

# find the mean for the columns and calculates the proportion of rows where the emotion match is TRUE, i.e. =1
alignment_rates = df[["Human_match", "FT_match", "GPT_match"]].mean()
print(alignment_rates)

# columns needed for this part of the analysis
sim_cols = ["Human_response_emotion_similarity",
            "FT_response_emotion_similarity",
            "GPT_response_emotion_similarity"]

# creating data frame for this part of it with summary stats
sim_summary = pd.DataFrame({
    "Mean": df[sim_cols].mean(),
    "Median": df[sim_cols].median(),
    "Min": df[sim_cols].min(),
    "Max": df[sim_cols].max(),
    "StdDev": df[sim_cols].std()
}).round(3)

print(sim_summary)

# extract the emotional similarity columns
# Higher value = the response is emotionally closer to the client’s emotion vector
human = df["Human_response_emotion_similarity"]
ft = df["FT_response_emotion_similarity"]
gpt = df["GPT_response_emotion_similarity"]


# Making paired Wilcoxon signed-rank tests, which:
# compare two related samples that are non-parametric (i.e. doesn’t assume normal distribution)
# works row-by-row
# Choosing Wilcoxon, as the emotional similarity scores are bounded between 0 and 1 → 
# not normally distributed, slightly skewed and paired for each message.
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


''' -------------------- emotion distribution comparison --------------------- '''
# Counts how many times each emotion appears, turns raw counts into proportions (percentages) and rounds to 3 decimals.
# + making it ready to plot
# Emotion labels in fixed order
top_columns = ["Context_top", "Human_response_top", "FT_response_top", "GPT_response_top"]
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

distributions = {}

for col in top_columns:
    print(f"\nValue counts for {col}:")
    dist = df[col].value_counts(normalize=True).round(3)
    print(df[col].value_counts(normalize=True).round(3))
    # reindex ensures missing emotions appear as 0
    distributions[col] = dist.reindex(emotion_labels, fill_value=0)
    

''' --------------- For plotting ----------------- '''

# Convert to dataframe
plot_df = pd.DataFrame(distributions, index=emotion_labels)

# Custom color palette
colors = {
    "Context_top": "#4893f0",        # blue
    "Human_response_top": "#f0c14a", # orange
    "FT_response_top": "#6dcc5e",    # green
    "GPT_response_top": "#d92e2e"    # red
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