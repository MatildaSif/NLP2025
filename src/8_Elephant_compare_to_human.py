"""
ELEPHANT Metrics Analysis

This script processes validation data to evaluate social sycophancy behaviors
of human, GPT, and fine-tuned (FT) model responses on OEQ items.

The aim of this analysis is to answer RQ3:
RQ3: Social Sycophancy
How do model-generated responses differ from human responses
on metrics of validation, indirectness, and framing?

What the script does:
1. Loads validation CSV files for human, GPT, and FT responses.
2. Merges the datasets into a single DataFrame ('final_df_sycophancy.csv').
3. Computes mean scores and 95% confidence intervals for each metric per model.
4. Generates a grouped bar plot comparing human, GPT, and FT responses.
5. Computes differences from the human baseline for GPT and FT models,
   along with confidence intervals for these differences.
6. Saves processed metrics, difference calculations, and plots to CSV and PNG files.

Input: Human_validation.csv, GPT_validation.csv, FT_validation.csv
Intermediate output: final_df_sycophancy.csv
Final output: elephant_metrics_results.csv, elephant_metrics_differences.csv, elephant_metrics_plot.png

"""

# ------------------------ Imports ------------------------
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import transformers
import pandas as pd
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from utils import load_data, create_csv

# ------------------------ Seaborn Style Configuration ------------------------
sns.set_context("paper")
sns.set(font_scale=2.2)
sns.set_style("white", {
    "font.family": "sans-serif",
    "font.serif": ['Helvetica'],
    "font.scale": 2.2
})
sns.set_style("ticks", {
    "xtick.major.size": 4,
    "ytick.major.size": 4
})


# ------------------------ Functions ------------------------

def get_color_mapping():
    """
    Define color mapping for different models.
    
    Returns:
        dict: mapping of model names to hex colors
    """
    color_map = {
        "Human_sychophancy": "#5396c5",  # blue
        "FT_sychophancy": "#ff9c47",     # orange
        "GPT_sychophancy": "#69bc69"     # green
    }
    return color_map


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

    # Make ID first column
    cols = final_df.columns.tolist()
    cols.insert(0, cols.pop(cols.index("ID")))
    final_df = final_df[cols]

    create_csv(final_df, "final_df_sycophancy.csv", files_path)
    return final_df


def apply_style(ax):
    """Apply consistent styling to matplotlib axes."""
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for yy in [0.2, 0.4, 0.6, 0.8]:  # dashed horizontal lines
        ax.axhline(y=yy, linestyle='--', color='black', linewidth=1, alpha=0.3)


# ------------------------ Main Execution ------------------------

def main():
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Load and merge data files
    data_file_list = ["Human_validation.csv", "GPT_validation.csv", "FT_validation.csv"]
    files_path = "data/"

    dfs = []
    for file in data_file_list:
        df = load_data(files_path, file)
        dfs.append(df)

    merge_dfs(dfs, files_path)  # creates df called final_df_sycophancy.csv
    df = pd.read_csv("data/final_df_sycophancy.csv")
    print(df.head())

    # Extract model and metric info from columns
    data = []
    for col in df:
        for metric in ['validation', 'indirectness', 'framing']:
            if metric in col:
                model = col.replace(metric + '_', '')
                values = pd.to_numeric(df[col], errors='coerce').dropna().astype(int).values
                mean = values.mean()
                std = 1.96 * scipy.stats.sem(values)
                data.append({'model': model, 'metric': metric, 'mean': mean, 'CI': std, 'col': col})
                break

    plot_df = pd.DataFrame(data)
    print(plot_df)
    
    # Save the main metrics results to CSV
    plot_df.to_csv("output/elephant_metrics_results.csv", index=False)
    print(f"\nMetrics saved to output/elephant_metrics_results.csv")
    print(f"Rows saved: {len(plot_df)}")

    # Get color mapping
    color_map = get_color_mapping()
    
    # Create grouped bar plot
    metrics = ['validation', 'indirectness', 'framing']
    models = plot_df['model'].unique()
    x = np.arange(len(metrics))
    width = 0.1

    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot grouped bars for each model within each metric
    for i, model in enumerate(models):
        print(i)
        model_df = plot_df[plot_df['model'] == model].set_index('metric').loc[metrics]
        
        # Get color for this model
        color = color_map.get(model, "#000000")  # Default to black if not found
        
        hatch = '\\' if i == 0 else None
        ax.bar(
            x + i * width,
            model_df['mean'],
            width,
            yerr=model_df['CI'],
            label=model,
            hatch=hatch,
            color=color
        )
    
    apply_style(ax)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([x.capitalize() for x in metrics])
    ax.set_ylabel("Mean Score")
    ax.set_title("ELEPHANT Metrics of Social Sycophancy on OEQ")

    ax.legend(
        bbox_to_anchor=(0.97, 1.05),
        loc='upper left',
        borderaxespad=0,
        fontsize=20
    )
    plt.tight_layout()
    plt.savefig("output/elephant_metrics_plot.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Compute differences from human baseline for both GPT and FT models
    human = (
        plot_df.query("model == 'Human_sychophancy'")[["metric", "mean", "CI"]]
        .rename(columns={"mean": "mean_human", "CI": "CI_human"})
    )
    
    # Check if human data exists
    if human.empty:
        print("\nWarning: No human baseline data found. Skipping difference calculations.")
        return
    
    # Process GPT model
    tmp_gpt = (
        plot_df.query("model == 'GPT_sychophancy'")
        .merge(human, on="metric", how="left", validate="m:1")
    )
    
    # Process FT model
    tmp_ft = (
        plot_df.query("model == 'FT_sychophancy'")
        .merge(human, on="metric", how="left", validate="m:1")
    )
    
    # Combine both models
    tmp = pd.concat([tmp_gpt, tmp_ft], ignore_index=True)
    
    # Check if any model data exists
    if tmp.empty:
        print("\nWarning: No GPT_sychophancy or FT_sychophancy data found. Skipping difference calculations.")
        return

    # Compute difference and CI for the difference
    # CI -> SE assuming 95% CI: CI = 1.96 * SE
    z = 1.96
    se_model = tmp["CI"] / z
    se_human = tmp["CI_human"] / z
    se_diff = np.sqrt(se_model**2 + se_human**2)

    tmp["mean_diff"] = tmp["mean"] - tmp["mean_human"]
    tmp["CI_diff"] = z * se_diff
    tmp["lower"] = tmp["mean_diff"] - tmp["CI_diff"]
    tmp["upper"] = tmp["mean_diff"] + tmp["CI_diff"]

    # Final results
    final_df = tmp[[
        "model", "metric", "mean_diff", "CI_diff", "lower", "upper",
        "mean", "CI", "mean_human", "CI_human", "col"
    ]].sort_values(["model", "metric"]).reset_index(drop=True)

    # Save difference metrics to output file
    output_path = "output/elephant_metrics_differences.csv"
    final_df.to_csv(output_path, index=False)
    print(f"\nDifference metrics saved to {output_path}")
    print(f"File exists: {os.path.exists(output_path)}")
    print(f"Rows saved: {len(final_df)}")
    
    # Print results
    print("\nModel Performance vs Human Baseline:")
    for _, r in final_df.iterrows():
        print(f"{r['model']:>17} | {r['metric']:<13} "
              f"Δ={r['mean_diff']:.2%} ± {r['CI_diff']:.2%} "
              f"[{r['lower']:.2%}, {r['upper']:.2%}]")


if __name__ == "__main__":
    main()