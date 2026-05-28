import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys

# Load dataset from the same directory as this script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "raw_skills.csv")

if not os.path.exists(csv_path):
    print(f"Error: raw_skills.csv not found in {script_dir}")
    print("Make sure raw_skills.csv is in the same folder as this script.")
    sys.exit(1)

data = pd.read_csv(csv_path)
data['skills_list'] = data['Skills'].str.lower().str.split()

# TF-IDF initialization
vectorizer = TfidfVectorizer(analyzer=lambda x: x)
tfidf_matrix = vectorizer.fit_transform(data['skills_list'])

# Get user input
raw_input = input("Enter your skills (comma-separated): ")
user_input = [s.strip().lower() for s in raw_input.split(",")]

# Warn about unrecognized skills
known_vocab = set(vectorizer.vocabulary_.keys())
unknown = [s for s in user_input if s not in known_vocab]
if unknown:
    print(f"\nWarning: These skills weren't recognized and will be ignored: {unknown}")

user_vector = vectorizer.transform([user_input])

# Cosine Similarity
similarity_scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
top_3_indices = np.argsort(similarity_scores)[-3:][::-1]

# Print results
print("\n--- Top 3 Career Path Matches ---\n")
for rank, index in enumerate(top_3_indices, 1):
    role = data.iloc[index]['Role']
    score = similarity_scores[index]
    print(f"Rank {rank}: {role} (Score: {score:.4f})")
    print("-" * 40)