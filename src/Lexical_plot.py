'''
----------------------------- Lexical plots ----------------------------
This script generates simple summary density-style plots for the lexical analysis. 
It is focused on the CTTR and MTLD. And for each metric, it plots one curve per text type:
- Context
- Human response
- GPT response
- Fine-tuned response

These curves are approximated using the mean ± SD for each group.
Output: PNG files saved to OUTPUT_DIR.
'''

''' ------------------------------ SETUP ------------------------------- '''
# loading packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Defining paths
SUMMARY_PATH = "/work/NLP2025/output/lexical_summary.csv"
OUTPUT_DIR = "output/" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the summary CSV - while excluding the label column
df = pd.read_csv(SUMMARY_PATH, index_col=0)

''' ------------------ Making plot and stats summary functions -------------------'''
def make_density_plot(metric_name, stats_type):
    """
    Create a density-style plot for one of metric at a time.

    stats_type: dict like
        {
            "Context": (mean, sd),
            "Human response": (mean, sd),
            "GPT response": (mean, sd),
            "FT response": (mean, sd),
        }
    """
    plt.figure(figsize=(8, 5))

    for label, (mean, sd) in stats_type.items():
        # Uses three points (mean - sd, mean, mean + sd) to get a smooth-ish curve
        values = pd.Series([mean - sd, mean, mean + sd])
        sns.kdeplot(
            values,
            label=f"{label} (Mean={mean:.2f}, SD={sd:.2f})"
        )

    plt.title(f"Density Plot: {metric_name}")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.legend()

    out_path = os.path.join(OUTPUT_DIR, f"{metric_name}_density.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

def get_stats_for_metric(metric_suffix):
    """
    Given a metric suffix like 'CTTR' or 'MTLD', pull out the corresponding
    rows for each text type and return a dict of (mean, sd).
    """
    row_names = {
        "Context":        f"Context_{metric_suffix}",
        "Human response": f"Human_response_{metric_suffix}",
        "GPT response":   f"GPT_response_{metric_suffix}",
        "FT response":    f"FT_response_{metric_suffix}",
    }

    stats = {}
    for label, row_name in row_names.items():
        row = df.loc[row_name]
        stats[label] = (row["Mean"], row["SD"])

    return stats

''' ------------------- Generating plots -------------------- '''

# Generate plots for each metric
for metric in ["CTTR", "MTLD"]:
    stats = get_stats_for_metric(metric)
    make_density_plot(metric, stats)

print("\nAll CTTR and MTLD density plots saved!")