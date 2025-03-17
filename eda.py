import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the datasets
movies_df = pd.read_csv("cleaned_movies.csv")
ratings_df = pd.read_csv("cleaned_ratings.csv")

# Exploratory Data Analysis (EDA)
print("Missing Values in Movies Dataset:\n", movies_df.isnull().sum())
print("Missing Values in Ratings Dataset:\n", ratings_df.isnull().sum())

# Visualizing Movie Ratings Distribution
plt.figure(figsize=(8, 5))
sns.histplot(ratings_df['rating'], bins=20, kde=True)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Top 10 Most Rated Movies
top_movies = ratings_df['movieId'].value_counts().head(10).reset_index()
top_movies.columns = ['movieId', 'Rating Count']
top_movies = top_movies.merge(movies_df[['movieId', 'title']], on='movieId')
print("Top 10 Most Rated Movies:\n", top_movies)

# Recommendation System with Surprise Library
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

# Train an SVD model
model = SVD()
model.fit(trainset)

# Evaluate the model
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# Predict for a specific user and movie
def recommend_movie(user_id, movie_id):
    prediction = model.predict(user_id, movie_id)
    return prediction.est

# Example Recommendation
user_id = 1
movie_id = top_movies['movieId'].iloc[0]  # Top-rated movie
predicted_rating = recommend_movie(user_id, movie_id)
print(f"Predicted Rating for User {user_id} for Movie '{top_movies['title'].iloc[0]}': {predicted_rating:.2f}")
