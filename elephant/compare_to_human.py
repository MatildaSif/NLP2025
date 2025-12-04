import pandas as pd
from functools import reduce


"""
Embeddings and data cleaning functions

This script contains functions for:
- Loading a tokeniser
- Loading a Hugging Face model
- Loading a data frame
- Creating a CSV from a dataframe

Dependencies:
- Pandas for data manipulation.
- Joblib for saving/loading models.
- Matplotlib for confusion matrix visualization.
"""

# ------------------------ Imports ------------------------
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import transformers
import pandas as pd



# ------------------------ Functions ------------------------

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
    print(f"Original data size: {len(df)}")
    return df


def create_csv(df, new_data_file, files_path):
    """
    Create csv from new data frame and save to data file path.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Context', 'ID' and 'Response' columns.
        new_data_file (str): Name of new csv file

    Returns:
        csv_data (csv): New saved csv file
    """
    df = df.copy()
    data_path = os.path.join(files_path, new_data_file)
    csv_data = df.to_csv(data_path, index = False)
    return csv_data



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

    #final_df = final_df[["ID","Context","Human_response", "FT_response", "GPT_response", "topic"]]
    # Make ID first column
    cols = final_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("ID")))
    final_df = final_df[cols]

    create_csv(final_df, "final_df_sycophancy.csv", files_path)
    return final_df


data_file_list = ["Human_validation.csv", "GPT_validation.csv", "FT_validation.csv"]
files_path = "../data/"

dfs = []
for file in data_file_list:
    df = load_data(files_path, file)
    dfs.append(df)

merge_dfs(dfs, files_path) # creates df called final_df_sycophancy.csv


df = pd.read_csv("data/final_df_sycophancy.csv")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
sns.set_context("paper")
sns.set(font_scale = 2.2)
sns.set_style("white", {
    "font.family": "sans-serif",
    "font.serif": ['Helvetica'],
    "font.scale": 2.2
})
sns.set_style("ticks", {"xtick.major.size": 4,
                        "ytick.major.size": 4})

def apply_style(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for yy in [0.2,0.4,0.6,0.8]: # change to wherever you want dashed lines
        ax.axhline(y=yy, linestyle='--', color='black', linewidth=1, alpha=0.3)
    
# Extract model and metric info from columns
data = []
for col in df:
    for metric in ['validation', 'indirectness', 'framing']:
        if metric in col:
            model = col.replace(metric + '_', '')
            values = pd.to_numeric(df[col], errors='coerce').dropna().astype(int).values
            mean = values.mean()
            std = 1.96*scipy.stats.sem(values)
            data.append({'model': model, 'metric': metric, 'mean': mean, 'CI': std, 'col':col})
            break



plot_df = pd.DataFrame(data)
plot_df



metrics = [ 'validation','indirectness','framing']

models = plot_df['model'].unique()
x = np.arange(len(metrics))
width = 0.1

fig, ax = plt.subplots(figsize=(15, 5))


# Plot grouped bars for each model within each metric
for i, model in enumerate(models):
    print(i)
    model_df = plot_df[plot_df['model'] == model].set_index('metric').loc[metrics]
     
#     ax.errorbar(df.Feature, model_df['mean'],m, linewidth=0, marker='o', ms=5,
#                 elinewidth=1, color=color, alpha=0.7)
    hatch = '\\' if i == 0 else None
    ax.bar(
    x + i*width,
    model_df['mean'],
    width,
    yerr=model_df['CI'],
    label=model,
    hatch=hatch
    )
apply_style(ax)
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels([x.capitalize() for x in metrics])
ax.set_ylabel("Mean Score")
ax.set_title("ELEPHANT Metrics of Social Sycophancy on OEQ")


ax.legend(    bbox_to_anchor=(0.97, 1.05
                             ),  # x shifted left from 1.01 → 0.95, y shifted up from 1 → 1.05
 loc='upper left', borderaxespad=0,fontsize=20)#columnspacing=0.5)
plt.tight_layout()
plt.show()



import numpy as np
import pandas as pd

# split out human baselines
human = (
    plot_df.query("model == 'human'")[["metric", "mean", "CI"]]
    .rename(columns={"mean": "mean_human", "CI": "CI_human"})
)
# join back to non-human rows
tmp = (
    plot_df.query("model == 'gpt4o'")
    .merge(human, on="metric", how="left", validate="m:1")
)

# compute difference and CI for the difference
# CI -> SE assuming 95% CI: CI = 1.96 * SE
z = 1.96
se_model = tmp["CI"] / z
se_human = tmp["CI_human"] / z
se_diff = np.sqrt(se_model**2 + se_human**2)

tmp["mean_diff"] = tmp["mean"] - tmp["mean_human"]
tmp["CI_diff"] = z * se_diff
tmp["lower"] = tmp["mean_diff"] - tmp["CI_diff"]
tmp["upper"] = tmp["mean_diff"] + tmp["CI_diff"]

# final rate
final_df = tmp[[
    "model", "metric", "mean_diff", "CI_diff", "lower", "upper",
    "mean", "CI", "mean_human", "CI_human", "col"  # keep extras if useful
]].sort_values(["model", "metric"]).reset_index(drop=True)

for _, r in final_df.iterrows():
    print(f"{r['model']:>8} | {r['metric']:<13} "
          f"Δ={r['mean_diff']:.2%} ± {r['CI_diff']:.2%} "
          f"[{r['lower']:.2%}, {r['upper']:.2%}]")
