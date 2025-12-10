"""
On-topicness Analysis

This script computes row-wise cosine similarity between contexts and multiple response types,
and generates histogram and heatmap plots to visualize semantic alignment.

Goals:
- Calculate mean embeddings and plot
- Calculate average cosine similarities
- Plot results
- Analyze distributions per topic

"""


''' Setup '''
import os
os.environ["HF_HOME"] = "/work/tf_cache"
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import umap
from scipy import stats



''' Functions '''

def load_topics(csv_path):
    """
    Load the topic column from the CSV file.
    
    Args:
        csv_path (str): path to Human_responses.csv
    
    Returns:
        np.array: array of topics
    """
    df = pd.read_csv(csv_path)
    
    if 'topic' not in df.columns:
        print(f"Available columns: {df.columns.tolist()}")
        raise KeyError("'topic' column not found in CSV")
    
    return df['topic'].values


def compute_cosine_similarity(save_path, column_pairs):
    """
    Compute cosine similarity per column pairs e.g. context_1 v. human_response_context_1

    Args:
        save_path (str): folder where .npy embeddings are stored
        column_pairs (list of tuples): pairs of embedding column names
                                       e.g., [("context", "human_response"), ...]
    
    Returns:
        dict: average cosine similarity per pair
    """
    avg_cos_sim = {}
    rowwise_sims = {}

    for col1, col2 in column_pairs:
        X1 = np.load(os.path.join(save_path, f"emb-{col1}.npy"))
        X2 = np.load(os.path.join(save_path, f"emb-{col2}.npy"))

        # Row-wise cosine similarity
        X1_norm = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
        X2_norm = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
        rowwise_sim = np.sum(X1_norm * X2_norm, axis=1) # row-wise similarity

        avg_cos_sim[f"{col1}-{col2}"] = np.mean(rowwise_sim) # get average
        rowwise_sims[f"{col1}-{col2}"] = rowwise_sim

    return avg_cos_sim, rowwise_sims # also get row-wise sim for later plotting


def get_color_mapping():
    """
    Define color mapping for different comparison pairs.
    
    Returns:
        dict: mapping of pair names to hex colors
    """
    color_map = {
        "context-human_response": "#5396c5",  # blue
        "context-ft_response": "#ff9c47",     # orange
        "context-gpt_response": "#69bc69"    # green
    }
    return color_map


def plot_kde(rowwise_sims, output_path, filename, topics=None):
    """
    Plot KDE (smoothed density) of row-wise cosine similarities.

    Args:
        rowwise_sims (dict): dictionary with row-wise cosine similarities
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved
        topics (np.array): optional array of topics for per-topic plots
    """
    os.makedirs(output_path, exist_ok=True)
    color_map = get_color_mapping()

    if topics is None:
        # Overall plot
        plt.figure(figsize=(8,5))
        for name, sims in rowwise_sims.items():
            color = color_map.get(name, "#000000")  # Default to black if not found
            sns.kdeplot(sims, label=name, fill=True, alpha=0.5, color=color)
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Density")
        plt.title("Row-wise Cosine Similarity Distribution (KDE)")
        plt.legend()
        plt.tight_layout()

        save_file = os.path.join(output_path, filename)
        plt.savefig(save_file)
        plt.close()
        print(f"KDE plot saved to: {save_file}")
    else:
        # Per-topic faceted plot
        unique_topics = np.unique(topics[~pd.isna(topics)])
        n_topics = len(unique_topics)
        
        # Calculate grid dimensions
        n_cols = min(3, n_topics)
        n_rows = int(np.ceil(n_topics / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_topics == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, topic in enumerate(unique_topics):
            topic_mask = topics == topic
            ax = axes[idx]
            
            for name, sims in rowwise_sims.items():
                topic_sims = sims[topic_mask]
                if len(topic_sims) > 0:
                    color = color_map.get(name, "#000000")
                    sns.kdeplot(topic_sims, label=name, fill=True, alpha=0.5, color=color, ax=ax)
            
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Density")
            ax.set_title(f"Topic: {topic}")
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_topics, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Cosine Similarity Distribution by Topic", fontsize=16, y=1.00)
        plt.tight_layout()

        save_file = os.path.join(output_path, filename)
        plt.savefig(save_file, bbox_inches='tight')
        plt.close()
        print(f"Faceted KDE plot saved to: {save_file}")


def calculate_similarity_stats(rowwise_sims, topics=None):
    """
    Calculate mean, SD, and 95% confidence intervals for cosine similarities.
    
    Args:
        rowwise_sims (dict): dictionary with row-wise cosine similarities
        topics (np.array): optional array of topics for per-topic statistics
    
    Returns:
        tuple: (overall_stats_df, topic_stats_df) or (overall_stats_df, None)
    """
    
    # Helper function to calculate stats
    def compute_stats(data):
        n = len(data)
        mean = np.mean(data)
        sd = np.std(data, ddof=1)
        se = stats.sem(data)
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
        
        return {
            'n': n,
            'mean': mean,
            'sd': sd,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    # Overall statistics
    overall_data = []
    for name, sims in rowwise_sims.items():
        stat_dict = compute_stats(sims)
        stat_dict['pair'] = name
        overall_data.append(stat_dict)
    
    overall_stats_df = pd.DataFrame(overall_data)
    overall_stats_df = overall_stats_df[['pair', 'n', 'mean', 'sd', 'ci_lower', 'ci_upper']]
    
    # Per-topic statistics
    if topics is not None:
        topic_data = []
        unique_topics = np.unique(topics[~pd.isna(topics)])
        
        for topic in unique_topics:
            topic_mask = topics == topic
            for name, sims in rowwise_sims.items():
                topic_sims = sims[topic_mask]
                if len(topic_sims) > 0:
                    stat_dict = compute_stats(topic_sims)
                    stat_dict['pair'] = name
                    stat_dict['topic'] = topic
                    topic_data.append(stat_dict)
        
        topic_stats_df = pd.DataFrame(topic_data)
        topic_stats_df = topic_stats_df[['topic', 'pair', 'n', 'mean', 'sd', 'ci_lower', 'ci_upper']]
        
        return overall_stats_df, topic_stats_df
    
    return overall_stats_df, None


def save_statistics(overall_stats, topic_stats, output_path, prefix):
    """
    Save statistics to CSV files.
    
    Args:
        overall_stats (pd.DataFrame): overall statistics dataframe
        topic_stats (pd.DataFrame): per-topic statistics dataframe (can be None)
        output_path (str): folder where files should be saved
        prefix (str): prefix for filename (e.g., 'Topic' or 'Human')
    """
    os.makedirs(output_path, exist_ok=True)
    
    # Save overall statistics
    overall_file = os.path.join(output_path, f"{prefix}_overall_statistics.csv")
    overall_stats.to_csv(overall_file, index=False)
    print(f"Overall statistics saved to: {overall_file}")
    
    # Save per-topic statistics
    if topic_stats is not None:
        topic_file = os.path.join(output_path, f"{prefix}_topic_statistics.csv")
        topic_stats.to_csv(topic_file, index=False)
        print(f"Per-topic statistics saved to: {topic_file}")


''' Define Parameters '''
save_path = "data/emb/"
output_path = "output/"
topics_csv_path = "data/Human_responses.csv"

# The following are all lowercase to reflect the naming of the embedding files!
column_pairs_topic = [
    ("context", "human_response"),
    ("context", "ft_response"),
    ("context", "gpt_response")
]


''' Main '''
if __name__ == "__main__":
    save_path = save_path
    output_path = output_path

    # Load topics
    print("Loading topics from CSV...")
    topics = load_topics(topics_csv_path)
    print(f"Loaded {len(topics)} rows with topics")
    print(f"Unique topics: {len(np.unique(topics[~pd.isna(topics)]))}")

    # --- On topic Analysis ---

    # Compute average cosine similarities
    avg_cos_sim, rowwise_sims = compute_cosine_similarity(save_path, column_pairs = column_pairs_topic)
    print("Average Cosine Similarities:", avg_cos_sim)
    
    # Overall plots
    plot_kde(rowwise_sims, output_path, filename ="Topic_cosine_similarity_kde.png")

    # Per-topic plots
    print("\nGenerating per-topic plots for on-topic analysis...")
    plot_kde(rowwise_sims, output_path, filename ="Topic_cosine_similarity_kde_by_topic.png", topics=topics)
    
    # Calculate and save statistics
    print("\nCalculating statistics for on-topic analysis...")
    overall_stats, topic_stats = calculate_similarity_stats(rowwise_sims, topics)
    save_statistics(overall_stats, topic_stats, output_path, prefix="Topic")
    print("\nOn-topic statistics summary:")
    print(overall_stats)