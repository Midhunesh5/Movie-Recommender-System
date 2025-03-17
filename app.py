import streamlit as st
import pandas as pd
import ast
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gc  # Garbage collector

# Configure Streamlit to improve performance
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Add caching to expensive operations
@st.cache_data
def load_data():
    try:
        # Load only required columns to reduce memory usage
        movies = pd.read_csv(
            "cleaned_dataset/cleaned_movies.csv", 
            usecols=['title', 'genres', 'keywords', 'overview']
        )
        
        # Filter out rows with missing titles
        movies = movies.dropna(subset=['title'])
        
        # Convert to lowercase and remove duplicates to reduce data size
        movies['title'] = movies['title'].str.lower()
        movies = movies.drop_duplicates(subset=['title']).reset_index(drop=True)
        
        # Fill NaN values with empty strings to avoid processing errors
        movies['genres'] = movies['genres'].fillna('[]')
        movies['keywords'] = movies['keywords'].fillna('[]')
        movies['overview'] = movies['overview'].fillna('')
        
        return movies
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return a minimal valid DataFrame to avoid crashing
        return pd.DataFrame({'title': ['sample_movie'], 'genres': ['[]'], 'keywords': ['[]'], 'overview': ['']})

# Create a more efficient processing pipeline with caching
@st.cache_data
def process_features(movies):
    # Process text features more efficiently
    def extract_text(text, default=''):
        if not isinstance(text, str) or text == '':
            return default
        try:
            items = ast.literal_eval(text)
            if isinstance(items, list):
                return ' '.join(items) if items else default
            return default
        except (ValueError, SyntaxError):
            return text
    
    # Apply extraction only to necessary columns
    genres = movies['genres'].apply(extract_text)
    keywords = movies['keywords'].apply(extract_text)
    
    # Clean overview - no need for ast.literal_eval
    overview = movies['overview'].fillna('')
    
    # Combine features directly without intermediate storage
    combined_features = genres + ' ' + overview + ' ' + keywords
    
    # Ensure no empty strings in the features
    combined_features = combined_features.apply(lambda x: x if x.strip() else 'unknown')
    
    return combined_features

# Cache the vectorization and similarity computation
@st.cache_resource
def compute_similarity(combined_features):
    # Make sure we have data to process
    if len(combined_features) == 0:
        st.error("No movie data available to process")
        # Return empty matrices to avoid crashing
        return np.array([[]]), None
    
    # Use fewer features and sparse matrices to reduce memory usage
    vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=3000,
        min_df=1,  # Changed from 2 to 1 to handle sparse data better
        max_df=0.95  # Increased from 0.85 to include more terms
    )
    
    try:
        # Make sure we have at least one non-empty string
        if combined_features.str.strip().str.len().sum() == 0:
            st.warning("Warning: All feature texts are empty. Using placeholder data.")
            # Create a single placeholder feature to avoid empty array errors
            combined_features = pd.Series(["placeholder_feature"])
        
        # Transform in smaller batches if dataset is large
        if len(combined_features) > 10000:
            # Process in chunks of 5000 movies
            chunk_size = 5000
            tfidf_matrix = None
            
            for i in range(0, len(combined_features), chunk_size):
                chunk = combined_features.iloc[i:i+chunk_size]
                
                # Only fit once on the first chunk, then transform only
                if i == 0:
                    chunk_tfidf = vectorizer.fit_transform(chunk)
                else:
                    chunk_tfidf = vectorizer.transform(chunk)
                
                if tfidf_matrix is None:
                    tfidf_matrix = chunk_tfidf
                else:
                    tfidf_matrix = np.vstack([tfidf_matrix, chunk_tfidf])
                    
                # Force garbage collection
                gc.collect()
        else:
            # Process all at once for smaller datasets
            tfidf_matrix = vectorizer.fit_transform(combined_features)
        
        return tfidf_matrix, vectorizer
    
    except Exception as e:
        st.error(f"Error in TF-IDF processing: {e}")
        # Create a minimal valid tfidf_matrix to avoid crashing
        return vectorizer.fit_transform(["placeholder"]), vectorizer

# Cache lookup dictionary creation
@st.cache_data
def create_indices(movies):
    return {title: idx for idx, title in enumerate(movies['title'])}

# Optimize recommendation function with error handling
def get_recommendations(selected_movie, movies, tfidf_matrix, movie_indices, top_n=10):
    selected_movie = selected_movie.lower()
    
    if not movie_indices or selected_movie not in movie_indices:
        return None, ["Movie not found. Please enter a valid movie name."]
    
    try:
        # Get the index of the movie
        idx = movie_indices[selected_movie]
        
        # Verify the index is valid
        if idx >= tfidf_matrix.shape[0]:
            return None, ["Error: Movie index out of bounds. Please try another movie."]
        
        # Only compute similarity for the selected movie, not the entire matrix
        movie_vector = tfidf_matrix[idx:idx+1]
        
        # Check if movie vector is empty or contains only zeros
        if movie_vector.nnz == 0:
            return None, ["Error: No features available for this movie. Please try another one."]
        
        # Calculate similarity with other movies (more memory efficient)
        sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
        
        # Create a simple list of (index, score) tuples
        movie_scores = list(enumerate(sim_scores))
        
        # Sort by similarity score and get top matches
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        # Make sure we have enough movies (exclude the selected movie)
        if len(movie_scores) <= 1:
            return None, ["Not enough similar movies found. Please try another title."]
            
        movie_scores = movie_scores[1:min(top_n+1, len(movie_scores))]  # Skip the movie itself
        
        # Get indices of selected movies
        movie_indices_list = [i[0] for i in movie_scores]
        
        # Get the selected movies with their scores
        result_df = movies.iloc[movie_indices_list][['title', 'genres']].copy()
        result_df['score'] = [score for _, score in movie_scores]
        
        # Format titles and genres for display
        result_df['title'] = result_df['title'].str.title()
        
        # More efficient genre formatting
        def format_genres(genres_str):
            if not isinstance(genres_str, str) or genres_str == '':
                return "Not specified"
            try:
                genres_list = ast.literal_eval(genres_str)
                if isinstance(genres_list, list) and genres_list:
                    return ", ".join(genres_list)
                return "Not specified"
            except (ValueError, SyntaxError):
                return str(genres_str)
        
        result_df['genres'] = result_df['genres'].apply(format_genres)
        
        return result_df, None
        
    except Exception as e:
        return None, [f"An error occurred: {str(e)}. Please try another movie."]

# Streamlit Interface with optimizations
def main():
    st.title("Movie Recommendation System")
    
    # Add a loading spinner
    with st.spinner("Loading movie data..."):
        # Load and process data with caching
        movies = load_data()
        
    # Create sidebar for status and info
    with st.sidebar:
        st.info(f"Database contains {len(movies)} movies")
        st.subheader("Performance Tips")
        st.markdown("""
        - First search might be slow (building index)
        - Subsequent searches will be much faster
        - Try exact movie titles for best results
        """)
    
    # Check if we have data
    if len(movies) == 0:
        st.error("No movie data available. Please check your dataset file.")
        return
    
    # Process features after database is loaded
    with st.spinner("Processing movie features..."):
        combined_features = process_features(movies)
    
    # Compute similarity with a progress indicator
    with st.spinner("Building recommendation engine (first-time only)..."):
        tfidf_matrix, vectorizer = compute_similarity(combined_features)
    
    # Check if matrix was created successfully
    if tfidf_matrix.shape[0] == 0:
        st.error("Failed to create recommendation engine. Please check your dataset.")
        return
    
    # Create indices for lookup
    movie_indices = create_indices(movies)
    
    # User input
    selected_movie = st.text_input("Enter a Movie Name")
    
    # Option to limit results for faster processing
    col1, col2 = st.columns([3, 1])
    with col2:
        top_n = st.slider("Number of recommendations", 5, 20, 10)
    
    if st.button("Get Recommendations") and selected_movie:
        with st.spinner("Finding similar movies..."):
            recommendations, error = get_recommendations(
                selected_movie, 
                movies, 
                tfidf_matrix, 
                movie_indices,
                top_n
            )
        
        if error:
            st.error(error[0])
            # Show closest matches as suggestions
            closest_matches = [title.title() for title in movie_indices.keys() 
                              if selected_movie.lower() in title][:5]
            if closest_matches:
                st.write("Did you mean one of these?")
                for match in closest_matches:
                    st.write(f"- {match}")
        else:
            st.subheader(f"Top recommendations based on '{selected_movie}':")
            
            # Format the similarity score as percentage
            recommendations['Similarity'] = recommendations['score'].apply(
                lambda x: f"{x*100:.1f}%"
            )
            
            # Display results without the raw score column
            st.dataframe(
                recommendations[['title', 'genres', 'Similarity']]
                .rename(columns={'title': 'Title', 'genres': 'Genres'})
            )

# Run the app with error handling
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.info("Please check your dataset file and try again.")