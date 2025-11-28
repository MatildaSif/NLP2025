""" 
Sentiment Classification 

Hugging Face Model Script This script applies a HF Model that will create a binary sentiment classification of our context and responses. 
1 = Positive sentiment 
0 = Negative sentiment 

Statistical analysis is then applied to assess how emotionally coherent each response is in relation to the context. 

Goals:
- Generate classification from a Hugging face model to real human mental health prompts 
- Analyse the classifications for emotional coherence - where sentiment is the same... 
""" 

''' Setup''' 
import os 
from transformers import pipeline 
from utils import load_data, create_csv 
from tqdm import tqdm

''' Functions'''
def sentiment_classification(pipeline, df, text_columns=None, save_path=None):
    if text_columns is None:
        text_columns = [col for col in df.columns if df[col].dtype == "object"]

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_size = 4

    for col in text_columns:
        print(f"Classifying column: {col}")
        pos_probs = []
        neg_probs = []

        texts = df[col].fillna("").tolist()

        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            results = pipeline(batch, truncation=True, max_length=512)

            for r in results:
                if r["label"].upper() == "POSITIVE":
                    prob_pos = r["score"]
                else:
                    prob_pos = 1 - r["score"]

                prob_neg = 1 - prob_pos

                pos_probs.append(prob_pos)
                neg_probs.append(prob_neg)

        df[f"prob_pos_{col.lower()}"] = pos_probs
        df[f"prob_neg_{col.lower()}"] = neg_probs

    if save_path is not None:
        df.to_csv(save_path, index=False)

    return df


'''Main''' 
if __name__ == "__main__": 
    ''' Parameters''' 
    model_name = "siebert/sentiment-roberta-large-english" 
    files_path = "data/" 
    data_file = "final_df.csv" 
    new_data_file = "final_df_sentiment.csv" 
    save_path = os.path.join(files_path, new_data_file) 
    text_columns = ["Context", "Human_response", "GPT_response", "FT_response"] 
    
    pipeline = pipeline("text-classification", model= model_name, truncation=True) # There is truncation so some longer than 512 tokens will be cut off at 512 tokens which is max.
    
    df = load_data(files_path, data_file) 
    df = sentiment_classification(pipeline, df, text_columns, save_path) 
    create_csv(df, new_data_file, files_path)
