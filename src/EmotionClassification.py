"""
EMOTION CLASSIFICATION
This script investigates whether responses are emotionally aligned with 
the client's emotional state.

For each row, it computes the emotion for both context and all response types.
Emotional coherence is computed as a continuous measure via cosine similarity.
"""


''' SETUP '''
# load packages
import pandas as pd
import numpy as np
from transformers import pipeline
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Needed so tqdm works with pandas .apply()
tqdm.pandas()

# load model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)
# defining paths
FINAL_DF_PATH = "/work/NLP2025/data/final_df.csv"
OUTPUT_PATH = "/work/NLP2025/data/emotion_analysis.csv"

# Load data
df = pd.read_csv(FINAL_DF_PATH)

# column names
emotion_cols = ["Context", "Human_response", "FT_response", "GPT_response"]

# labels for vector representation
emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Load emotion classifier
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

''' --------------- Define functions --------------- '''
# function for computing the classification using the huggingface pipeline
def get_emotion_scores(text):
    """Return emotion probabilities from the model."""
    if not isinstance(text, str) or text.strip() == "":
        return None
    scores = emotion_classifier(text, truncation=True)[0]
    return {d['label']: d['score'] for d in scores}

# functions for computing emotional alignment
def emotion_vector(score_dict):
    """Convert dict of label:score into a fixed vector matching emotion_labels."""
    return np.array([score_dict.get(e, 0) for e in emotion_labels])

def emotional_alignment(row, col):
    """Compute emotional similarity between context and a response."""
    ctx_vec = emotion_vector(row["Context_emotion"])
    resp_vec = emotion_vector(row[f"{col}_emotion"])
    return 1 - cosine(ctx_vec, resp_vec)


''' Main '''
if __name__ == "__main__":

    # compute emotion for all text columns
    for col in emotion_cols:
        print(f"Computing emotion scores for: {col}")
        df[f"{col}_emotion"] = df[col].progress_apply(get_emotion_scores)

    # compute emotional similarity
    for col in ["Human_response", "FT_response", "GPT_response"]:
        print(f"Computing emotional similarity for: {col}")
        df[f"{col}_emotion_similarity"] = df.progress_apply(
            lambda r: emotional_alignment(r, col), 
            axis=1
        )

    # Save emotional analysis results
    df.to_csv(OUTPUT_PATH, index=False)
