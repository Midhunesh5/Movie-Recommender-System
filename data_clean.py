import pandas as pd
import numpy as np
import ast

# File paths
data_path = "dataset"

# Load datasets
movies = pd.read_csv(f"{data_path}/movies_metadata.csv", low_memory=False)
ratings = pd.read_csv(f"{data_path}/ratings.csv")
credits = pd.read_csv(f"{data_path}/credits.csv")
keywords = pd.read_csv(f"{data_path}/keywords.csv")

# 1. Clean movies_metadata.csv
movies = movies.drop_duplicates(subset=['id'])
movies = movies[movies['id'].str.isnumeric()]  # Remove non-numeric IDs
movies['id'] = movies['id'].astype(int)
movies = movies[['id', 'title', 'genres', 'overview', 'release_date']]

# Convert genres from string to list
def parse_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres] if isinstance(genres, list) else []
    except:
        return []

movies['genres'] = movies['genres'].apply(parse_genres)
movies.dropna(subset=['title'], inplace=True)  # Remove rows without titles

# Handle missing overview and release_date
movies['overview'] = movies['overview'].fillna("No overview available.")
movies['release_date'] = movies['release_date'].fillna("0000-00-00")
movies = movies[movies['release_date'].str.match(r'\d{4}-\d{2}-\d{2}')]  # Keep valid dates

# 2. Clean ratings.csv
ratings.drop_duplicates(inplace=True)
ratings = ratings[['userId', 'movieId', 'rating']]

# Filter out inactive users
user_counts = ratings['userId'].value_counts()
active_users = user_counts[user_counts >= 5].index
ratings = ratings[ratings['userId'].isin(active_users)]

# 3. Clean credits.csv
def extract_names(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        return [person['name'] for person in crew] if isinstance(crew, list) else []
    except:
        return []

credits['crew'] = credits['crew'].apply(extract_names)
credits['cast'] = credits['cast'].apply(extract_names)
credits.dropna(subset=['cast', 'crew'], inplace=True)

# 4. Clean keywords.csv
keywords['keywords'] = keywords['keywords'].apply(parse_genres)

# 5. Merge datasets
movies = movies.merge(credits, left_on='id', right_on='id', how='left')
movies = movies.merge(keywords, left_on='id', right_on='id', how='left')

# Save cleaned files
movies.to_csv(f"{data_path}/cleaned_movies.csv", index=False)
ratings.to_csv(f"{data_path}/cleaned_ratings.csv", index=False)

print("Data cleaning completed. Cleaned files saved!")