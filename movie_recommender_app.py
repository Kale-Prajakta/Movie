import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

st.title("🎬 Content-Based Movie Recommender")

# Preprocess genres
movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['genres'] = movies['genres'].fillna('')

# Display genre distribution
movies['num_genres'] = movies['genres'].apply(lambda x: len(x.split()))
st.subheader("Distribution of Number of Genres")
fig1, ax1 = plt.subplots()
sns.histplot(movies['num_genres'], bins=20, ax=ax1)
st.pyplot(fig1)

# TF-IDF Vectorization on genres
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute cosine similarity
cos_sim = cosine_similarity(tfidf_matrix)

# Recommender UI
st.subheader("🎯 Recommend Similar Movies Based on Genre")
movie_titles = movies['title'].tolist()
selected_movie = st.selectbox("Choose a movie:", movie_titles)

if selected_movie:
    idx = movies[movies['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[1:6]]
    top_movies = movies.iloc[top_indices]['title'].tolist()

    st.write("Top 5 Recommendations:")
    for i, title in enumerate(top_movies, 1):
        st.write(f"{i}. {title}")
