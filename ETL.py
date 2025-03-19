import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('data/TMDB_movie_dataset_v11.csv', low_memory=False)

# 1. Eliminar columnas irrelevantes
cols_to_drop = ['homepage', 'backdrop_path', 'poster_path', 'tagline', 'imdb_id']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 2. Manejo de valores nulos
df['budget'].fillna(df['budget'].median(), inplace=True)
df['revenue'].fillna(df['revenue'].median(), inplace=True)
df['runtime'].fillna(df['runtime'].median(), inplace=True)

df.dropna(subset=['title'], inplace=True)  # Eliminar si falta información clave

# 3. Filtrar películas con menos de 10 votos
df = df[df['vote_count'] >= 10]

# 4. Normalización de 'vote_average'
df['vote_average_norm'] = df['vote_average'] / 10  # Escalar entre 0 y 1

# 5. Extraer año de 'release_date'
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Guardar dataset limpio
df.to_csv('peliculas_procesadas.csv', index=False)

# 6. Visualización con Streamlit
st.title("Análisis y Transformación del Dataset de Películas")

# Histograma de calificaciones
st.subheader("Distribución de Calificaciones Antes y Después de Normalización")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['vote_average'], bins=20, kde=True, ax=ax[0])
ax[0].set_title("Antes de Normalización")
sns.histplot(df['vote_average_norm'], bins=20, kde=True, ax=ax[1])
ax[1].set_title("Después de Normalización")
st.pyplot(fig)

# Mostrar dataset final
st.subheader("Dataset Limpio y Transformado")
st.dataframe(df.head())

# Guardamos el nuevo Dataset
df.to_csv('data/peliculas_procesadas.csv', index=False)

# Mostrar el schema final
st.subheader("Schema Final del Dataset Procesado")
st.text(", ".join(df.columns))

