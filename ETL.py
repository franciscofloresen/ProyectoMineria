import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import ast  # Para convertir strings en listas de Python si vienen en formato de texto

# Cargar el dataset
df = pd.read_csv('data/TMDB_movie_dataset_v11.csv')

# 1. Eliminar columnas irrelevantes
cols_to_drop = ['homepage', 'backdrop_path', 'poster_path', 'tagline', 'imdb_id']
df.drop(columns=cols_to_drop, inplace=True)

# 2. Manejo de valores nulos
df['budget'].fillna(df['budget'].median(), inplace=True)
df['revenue'].fillna(df['revenue'].median(), inplace=True)
df['runtime'].fillna(df['runtime'].median(), inplace=True)

df.dropna(subset=['title', 'genres'], inplace=True)  # Eliminar si falta información clave

# 3. Filtrar películas con menos de 10 votos
df = df[df['vote_count'] >= 10]

# 4. Normalización de 'vote_average'
df['vote_average_norm'] = df['vote_average'] / 10  # Escalar entre 0 y 1

# 5. One-hot encoding para 'genres'
# Asegurar que la columna 'genres' sea una lista real y no un string
df['genres'] = df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])

# Crear un conjunto único de géneros
unique_genres = set()
df['genres'].apply(lambda x: unique_genres.update([genre.strip() for genre in x]))

# Aplicar one-hot encoding para los géneros
for genre in unique_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

# Eliminar la columna original 'genres' después de la transformación
df.drop(columns=['genres'], inplace=True)

# 6. Extraer año de 'release_date'
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Guardar dataset limpio
df.to_csv('peliculas_procesadas.csv', index=False)

# 7. Visualización con Streamlit
st.title("Análisis y Transformación del Dataset de Películas")

# Histograma de calificaciones
st.subheader("Distribución de Calificaciones Antes y Después de Normalización")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['vote_average'], bins=20, kde=True, ax=ax[0])
ax[0].set_title("Antes de Normalización")
sns.histplot(df['vote_average_norm'], bins=20, kde=True, ax=ax[1])
ax[1].set_title("Después de Normalización")
st.pyplot(fig)

# Gráfico de distribución de géneros
st.subheader("Distribución de Géneros")
genre_counts = df[list(unique_genres)].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=ax)
plt.xticks(rotation=45)
plt.title("Frecuencia de Películas por Género")
st.pyplot(fig)

# Mostrar dataset final
st.subheader("Dataset Limpio y Transformado")
st.dataframe(df.head())
df

# Guardamos el nuevo Dataset
df.to_csv('data/peliculas_procesadas.csv', index=False)