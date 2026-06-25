# Skill-Based Career Recommendation Engine

A lightweight Python script that uses NLP and machine learning to recommend the top 3 career paths based on a user's skills. It utilizes TF-IDF vectorization and Cosine Similarity to find the closest matches from a dataset.

## Features

* **Skill Vectorization:** Uses `TfidfVectorizer` to convert skill lists into numerical features.
* **Smart Matching:** Calculates `cosine_similarity` between user input and career roles.
* **Input Validation:** Automatically flags and warns users about unrecognized skills.
* **Ranked Output:** Displays the top 3 closest career matches with their respective similarity scores.

## Prerequisites

Ensure you have Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn

```

## Project Structure

```text
├── Recommendation_Engine.py  # Main Python script
└── raw_skills.csv             # Dataset containing 'Role' and 'Skills' columns

```

> **Note:** The script expects `raw_skills.csv` to be in the same directory. The CSV must contain at least two columns: **Role** and **Skills**.

## Usage

1. Run the script from your terminal:
```bash
python Recommendation_Engine.py

```


2. Enter your skills when prompted (comma-separated):
```text
Enter your skills (comma-separated): python, machine learning, pandas

```


3. View your ranked career matches.
