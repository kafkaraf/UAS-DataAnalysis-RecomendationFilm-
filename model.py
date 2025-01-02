#!/usr/bin/env python
# coding: utf-8

# # UAS - Data Analysis (IS388-B-EN) Lec & Lab 
# ### Rafi Rabbani - 60029

# ## Data Preparation

# In[110]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib


# In[111]:


movies=pd.read_csv('dataset_film.csv')
movies.head(10)


# In[112]:


movies.describe()


# In[113]:


movies.info()


# In[114]:


movies.isnull().sum()


# In[115]:


movies.columns


# In[116]:


movies=movies[['title', 'genres', 'release_year']]


# In[117]:


movies


# In[118]:


movies = movies.dropna()
movies.dropna(inplace=True)
movies


# In[119]:


categorical_columns = ['genres', 'release_year', 'title']
movies.loc[:, 'tags'] = movies[categorical_columns].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)


# In[120]:


movies


# In[121]:


categorical_columns = ['genres', 'release_year', 'title']
new_data = movies.drop(columns=categorical_columns)


# In[122]:


print(movies.head(10))


# In[128]:


# Boxplot distribusi release_year
plt.figure(figsize=(10, 6))
plt.boxplot(movies['release_year'], vert=True, patch_artist=True, boxprops=dict(facecolor='lightgreen', color='green'), whiskerprops=dict(color='green'))
plt.title('Boxplot of Release Year Distribution (Vertical)', fontsize=14)
plt.ylabel('Release Year', fontsize=12)
plt.show()


# In[130]:


# Memecah kolom 'genres' menjadi genre individual
genres_list = movies['genres'].str.split(';', expand=True).stack().reset_index(drop=True)

# Menghitung jumlah kemunculan setiap genre
genre_counts = genres_list.value_counts()

# Membuat histogram
plt.figure(figsize=(12, 8))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Movie Genres', fontsize=14)
plt.xlabel('Genres', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=90)
plt.show()


# ## Mengubah Data menjadi Representasi Vektor

# In[123]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray()
print(vector.shape)


# ## Membangun Model KNN

# In[21]:


# Membangun model KNN
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(vector)


# In[23]:


def recommend_knn(movie_title):
    # Normalisasi judul film yang dimasukkan
    movie_title = movie_title.strip().lower()

    # Cari film berdasarkan judul
    movie_idx = movies[movies['title'].str.lower().str.contains(movie_title)].index

    # Cek jika film ditemukan
    if movie_idx.empty:
        print(f"Movie '{movie_title}' not found.")
        return

    movie_idx = movie_idx[0]

    # Hitung nearest neighbors
    distances, indices = knn.kneighbors([vector[movie_idx]])

    print(f"Rekomendasi film untuk '{movie_title}':")
    for i in range(1, len(indices[0])):
        print(movies.iloc[indices[0][i]]['title'])


# In[47]:


# Contoh rekomendasi film
recommend_knn("Jumanji")


# ## Evaluasi Model

# In[48]:


from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Fungsi untuk mengonversi hasil rekomendasi menjadi binary relevance (1 jika relevan, 0 jika tidak)
def evaluate_recommendations(actual, predicted, k=5):
    actual_set = set(actual[:k])  # Ambil top-k relevan
    predicted_set = set(predicted[:k])  # Ambil top-k hasil rekomendasi

    # Precision, Recall, F1 Score Calculation
    tp = len(actual_set.intersection(predicted_set))  # True Positives
    fp = len(predicted_set - actual_set)  # False Positives
    fn = len(actual_set - predicted_set)  # False Negatives

    # Precision, Recall, and F1 Score
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1}

# Contoh data (film yang benar-benar relevan dan hasil rekomendasi dari model)
actual_relevant_movies = ["Star Wars", "The Empire Strikes Back", "Return of the Jedi"]  # Top-3 relevan film
recommended_movies = ["Star Wars", "The Force Awakens", "Revenge of the Sith", "Return of the Jedi", "Rogue One"]  # Hasil rekomendasi dari KNN

# Evaluasi model dengan k = 5
evaluation_results = evaluate_recommendations(actual_relevant_movies, recommended_movies, k=5)
print(f"Precision: {evaluation_results['precision']}")
print(f"Recall: {evaluation_results['recall']}")
print(f"F1 Score: {evaluation_results['f1']}")


# In[49]:


# Data
metrics = ['Precision', 'Recall', 'F1 Score']
values = [0.4, 0.6666666666666666, 0.5]

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(metrics, values, color=['blue', 'green', 'orange'])
plt.ylim(0, 1)
plt.title('Model Evaluation Metrics', fontsize=14)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.show()


# In[132]:


joblib.dump(knn, 'knn_model.joblib')


# In[133]:


joblib.dump(cv, 'cv_model.joblib')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




