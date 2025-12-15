#!/usr/bin/bash


# activate the environment
source ./env/bin/activate

# run the script
python src/DataCleaning.py
python src/1_Lexical_analysis.py
python src/2_Lexical_plot.py
python src/3_Embeddings.py
python src/4_Context_CosineSimilarities.py
python src/5_EmotionClassification.py
python src/6_EmotionStats.py
python src/8_Elephant_compare_to_human.py
  
# close the environment
deactivate