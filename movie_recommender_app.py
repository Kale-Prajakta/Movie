import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt



# Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

st.title("🎬 Movie Recommender System")

# Preprocess genres
movies['genres'] = movies['genres'].str.replace('|', ' ')
movies['num_genres'] = movies['genres'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)

# Display genre distribution
st.subheader("Distribution of Number of Genres")
sns.histplot(movies['num_genres'], bins=20)
st.pyplot()

# TF-IDF + average rating
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))

avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']
movies = movies.merge(avg_ratings, on='movieId', how='left')

features = np.hstack((tfidf_matrix.toarray(), movies[['avg_rating']].fillna(0).values))

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# PCA + Clustering
pca = PCA(n_components=2)
pca_features = pca.fit_transform(features_scaled)

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(features_scaled)
movies['Cluster'] = clusters

# Plot clusters
st.subheader("PCA Clustering of Movies")
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap='tab10')
st.pyplot()

# Cosine similarity
cos_sim = cosine_similarity(tfidf_matrix)

# Recommender
st.subheader("🎯 Recommend Movies")
movie_titles = movies['title'].tolist()
selected_movie = st.selectbox("Choose a movie:", movie_titles)

if selected_movie:
    idx = movies[movies['title'] == selected_movie].index[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_movies = [movie_titles[i[0]] for i in sim_scores[1:6]]

    st.write("Top 5 Recommendations:")
    for i, title in enumerate(top_movies, 1):
        st.write(f"{i}. {title}")
