from . import app
from flask import render_template, request
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
import difflib
import os

csv_folder = os.path.join(os.path.dirname(__file__), 'content')

movies = pd.read_csv(os.path.join(csv_folder, 'tmdb_5000_movies.csv'))
credits = pd.read_csv(os.path.join(csv_folder, 'tmdb_5000_credits.csv'))

movies = movies.merge(credits, on='title')

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.isnull().sum()
movies.dropna(inplace=True)
movies.isnull().sum()

movies['genres'] = movies['genres'].apply(ast.literal_eval)
movies['keywords'] = movies['keywords'].apply(ast.literal_eval)
movies['cast'] = movies['cast'].apply(ast.literal_eval)
movies['crew'] = movies['crew'].apply(ast.literal_eval)

movies['genres'] = movies['genres'].apply(lambda x: [i['name'] for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i['name'] for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in x][:3])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'][:1])

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

def remove_space(L):
    return [i.replace(" ", "") for i in L]

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

nltk.download('punkt')
ps = PorterStemmer()

def stems(text):
    return " ".join([ps.stem(i) for i in text.split()])

movies['tags'] = movies['tags'].apply(stems)

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vector)

@app.route('/')
@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')

@app.route('/movies')
def movies_page():
    return render_template('movies.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        movie_name = request.form['movie_name']
        list_of_all_titles = movies['title'].tolist()
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        index = movies[movies['title'] == find_close_match[0]].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movies = [movies.iloc[i[0]]['title'] for i in distances[1:6]]
        return render_template('recommendations.html', movie_name=movie_name, recommendations=recommended_movies)

    # Render the form if it's a GET request
    return render_template('movies.html')
  
