"""
On-topicness Analysis

This script computes row-wise cosine similarity between contexts and multiple response types,
and generates histogram and heatmap plots to visualize semantic alignment.

Goals:
- Calculate mean embeddings and plot
- Calculate average cosine similarities
- Plot results

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



''' Functions '''

def compute_average_cosine_similarity(save_path, column_pairs):
    """
    Compute averaged cosine similarity per column pairs e.g. context_1 v. human_response_context_1

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

    
def plot_average_distances(avg_cos_sim, output_path, filename):
    """
    Plot averaged cosine distances

    Args:
        avg_cos_sim (dict): dictionary with the average cosine similarities 
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved
    """

    # Ensure save directory exists
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(avg_cos_sim.keys(), avg_cos_sim.values(), color=["blue", "pink", "red"])
    plt.ylabel("Average Cosine Similarity")
    plt.title("Average Cosine Similarity Across Column Pairs")
    plt.xticks(rotation=30)
    plt.tight_layout()
    
    save_file = os.path.join(output_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"Average cosine distance plot saved to: {save_file}")



def plot_mean_embeddings(save_path, columns, output_path, filename):
    """
    Plot mean embeddings as 2D points after PCA.

    Args:
        save_path (str): folder where .npy embeddings are stored
        columns (str): list of columns that will a mean will be calculated for
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved
    """
    # Ensure save directory exists
    os.makedirs(output_path, exist_ok=True)

    # get mean embeddings of each column
    mean_embeds = {}
    for col in columns:
        X = np.load(os.path.join(save_path, f"emb-{col}.npy"))
        mean_embeds[col] = X.mean(axis=0)

    labels = list(mean_embeds.keys())
    X_mean = np.stack(list(mean_embeds.values()))

    # PCA to 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_mean)

    # Plot points
    plt.figure(figsize=(6,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], color=["blue","green","orange","red"], s=100)

    # Add labels
    for i, label in enumerate(labels):
        plt.text(X_2d[i,0]+0.01, X_2d[i,1]+0.01, label, fontsize=12)

    plt.title("Mean Embeddings Projected to 2D (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    
    save_file = os.path.join(output_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"Average cosine similarity plot saved to: {save_file}")


def plot_histogram(rowwise_sims, output_path, filename):
    """
    Plot histogram of row-wise cosine similarities

    Args:
        rowwise_sim (dict): dictionary with the row-wise cosine similarities 
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved

    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(8,5))
    for name, sims in rowwise_sims.items():
        plt.hist(sims, bins=50, alpha=0.5, label=name)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Distribution of Context-Response Cosine Similarities")
    plt.legend()
    plt.tight_layout()

    save_file = os.path.join(output_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"Histogram of Context-Response Cosine Similarities plot saved to: {save_file}")





def plot_kde(rowwise_sims, output_path, filename):
    """
    Plot KDE (smoothed density) of row-wise cosine similarities.

    Args:
        rowwise_sims (dict): dictionary with row-wise cosine similarities
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved
    """
    os.makedirs(output_path, exist_ok=True)

    plt.figure(figsize=(8,5))
    for name, sims in rowwise_sims.items():
        sns.kdeplot(sims, label=name, fill=True, alpha=0.4)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Row-wise Cosine Similarity Distribution (KDE)")
    plt.legend()
    plt.tight_layout()

    save_file = os.path.join(output_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"KDE plot saved to: {save_file}")


def plot_violin(rowwise_sims, output_path, filename):
    """
    Plot violin plot of row-wise cosine similarities per column pair.

    Args:
        rowwise_sims (dict): dictionary with row-wise cosine similarities
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved
    """
    os.makedirs(output_path, exist_ok=True)

    # Convert to long-format dataframe
    data = []
    for name, sims in rowwise_sims.items():
        for val in sims:
            data.append({"pair": name, "similarity": val})
    df = pd.DataFrame(data)

    plt.figure(figsize=(8,5))
    sns.violinplot(x="pair", y="similarity", data=df, inner="quartile")
    plt.title("Row-wise Cosine Similarity per Column Pair (Violin Plot)")
    plt.ylabel("Cosine Similarity")
    plt.xlabel("Column Pair")
    plt.xticks(rotation=30)
    plt.tight_layout()

    save_file = os.path.join(output_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"Violin plot saved to: {save_file}")


def plot_heatmap(rowwise_sims, output_path, filename):
    """
    Plot heatmap of row-wise cosine similarities.

    Args:
        rowwise_sims (dict): dictionary with row-wise cosine similarities
        output_path (str): folder where plots should be saved
        filename (str): Name of file to be saved
    """
    os.makedirs(output_path, exist_ok=True)

    sims_matrix = np.stack(list(rowwise_sims.values()), axis=1)  # shape: (n_rows, n_pairs)
    labels = list(rowwise_sims.keys())

    plt.figure(figsize=(10,6))
    sns.heatmap(sims_matrix, cmap="viridis", yticklabels=False, xticklabels=labels)
    plt.title("Row-wise Cosine Similarity Heatmap")
    plt.xlabel("Column Pair")
    plt.ylabel("Context Index")
    plt.tight_layout()

    save_file = os.path.join(output_path, filename)
    plt.savefig(save_file)
    plt.close()
    print(f"Heatmap saved to: {save_file}")





''' Define Parameters '''
save_path = "data/emb/"
output_path = "output/"
column_pairs_topic = [
    ("context", "human_response"),
    ("context", "ft_response"),
    ("context", "response")
]
column_pairs_human = [
    ("ft_response", "response"),
    ("human_response", "ft_response"),
    ("human_response", "response")
]

''' Main '''
if __name__ == "__main__":
    save_path = save_path
    output_path = output_path

    # --- On topic Analysis ---

    # Compute average cosine similarities
    avg_cos_sim, rowwise_sims = compute_average_cosine_similarity(save_path, column_pairs = column_pairs_topic)
    print("Average Cosine Similarities:", avg_cos_sim)

    # Plot average distances
    plot_average_distances(avg_cos_sim, output_path, filename = "topic_avg_cosine_distance.png")

    # Plot mean embeddings after PCA
    columns = ["context", "human_response", "ft_response", "response"]
    plot_mean_embeddings(save_path, columns, output_path, filename = "topic_mean_embeddings_PCA.png")

    # Plot
    plot_histogram(rowwise_sims, output_path, filename = "topic_cosine_similarity_histogram.png")
    plot_kde(rowwise_sims, output_path, filename ="topic_cosine_similarity_kde.png")
    plot_violin(rowwise_sims, output_path, filename ="topic_cosine_similarity_violin.png")
    plot_heatmap(rowwise_sims, output_path, filename ="topic_cosine_similarity_heatmap.png")



    # --- Human-Likeness Analysis ---
    
    # Compute average cosine similarities
    avg_cos_sim, rowwise_sims = compute_average_cosine_similarity(save_path, column_pairs = column_pairs_human)
    print("Average Cosine Similarities:", avg_cos_sim)

    # Plot average distances
    plot_average_distances(avg_cos_sim, output_path, filename = "human_avg_cosine_distance.png")

    # Plot mean embeddings after PCA
    columns = ["context", "human_response", "ft_response", "response"]
    plot_mean_embeddings(save_path, columns, output_path, filename = "human_mean_embeddings_PCA.png")

    # Plot
    plot_histogram(rowwise_sims, output_path, filename = "human_cosine_similarity_histogram.png")
    plot_kde(rowwise_sims, output_path, filename ="human_cosine_similarity_kde.png")
    plot_violin(rowwise_sims, output_path, filename ="human_cosine_similarity_violin.png")
    plot_heatmap(rowwise_sims, output_path, filename ="human_cosine_similarity_heatmap.png")