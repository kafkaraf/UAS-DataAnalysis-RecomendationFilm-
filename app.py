import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Load models
knn = joblib.load('knn_model.joblib')
cv = joblib.load('cv_model.joblib')

# Load the movies dataset
movies = pd.read_csv('dataset_film.csv')

# Preprocess data
categorical_columns = ['genres', 'release_year', 'title']
movies['tags'] = movies[categorical_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
new_data = movies.drop(columns=categorical_columns)
vector = cv.transform(new_data['tags'].values.astype('U')).toarray()

# Function to recommend movies
def recommend_knn(movie_title):
    movie_title = movie_title.strip().lower()

    # Find the movie index
    movie_idx = movies[movies['title'].str.lower().str.contains(movie_title)].index

    if movie_idx.empty:
        st.write(f"Movie '{movie_title}' not found.")
        return []

    movie_idx = movie_idx[0]

    # Find nearest neighbors
    distances, indices = knn.kneighbors([vector[movie_idx]])

    recommended_titles = []
    for i in range(1, len(indices[0])):
        recommended_titles.append(movies.iloc[indices[0][i]]['title'])

    return recommended_titles

# Streamlit interface
st.title('Movie Recommendation System')
st.markdown("This system recommends movies based on a given movie title.")

# User input for movie title
movie_input = st.text_input("Enter a movie title")

if st.button("Recommend Movies"):
    if movie_input:
        recommended_movies = recommend_knn(movie_input)
        if recommended_movies:
            st.write(f"Recommended movies similar to '{movie_input}':")
            for movie in recommended_movies:
                st.write(movie)
        else:
            st.write("No recommendations available.")
    else:
        st.write("Please enter a movie title.")
