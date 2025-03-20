import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_csv('data/TMDB_movie_dataset_v11.csv', low_memory=False)

# 1. Eliminar columnas irrelevantes
cols_to_drop = ['homepage', 'backdrop_path', 'poster_path', 'tagline', 'imdb_id', 'original_language', 'original_title', 'overview',  'production_countries', 'spoken_languages', 'keywords']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 2. Manejo de valores nulos
df['budget'].fillna(df['budget'].median(), inplace=True)
df['revenue'].fillna(df['revenue'].median(), inplace=True)
df['runtime'].fillna(df['runtime'].median(), inplace=True)
df.dropna(subset=['title'], inplace=True)  # Eliminar si falta información clave

# 3. Filtrar películas entre 2000 y 20000 votos
df = df[(df['vote_count'] >= 2000) & (df['vote_count'] <= 20000)]


# 4. Extraer año de 'release_date'
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year


# 5. Filtrar películas desde el año 2000
df = df[df['release_year'] >= 2000]  


# Guardar dataset limpio
df.to_csv('peliculas_procesadas.csv', index=False)

# Configuración de Streamlit
st.title("Exploración de Datos de Películas")
st.sidebar.header("Filtros")

# Filtros interactivos
año_min, año_max = int(df['release_year'].min()), int(df['release_year'].max())
selected_year = st.sidebar.slider("Selecciona un año de estreno", año_min, año_max, año_max)

genres = df['genres'].dropna().unique().tolist()
selected_genre = st.sidebar.selectbox("Selecciona un género", ["Todos"] + genres)

# Filtrar datos según la selección
df_filtered = df[df['release_year'] == selected_year]
if selected_genre != "Todos":
    df_filtered = df_filtered[df_filtered['genres'].str.contains(selected_genre, na=False)]

# Mostrar dataset filtrado
st.subheader(f"Películas estrenadas en {selected_year}")
st.dataframe(df_filtered[['title', 'vote_average', 'vote_count', 'release_date', 'budget', 'revenue', 'genres']].head(20))

# Gráfica de distribución de calificaciones
st.subheader("Distribución de Calificaciones")
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df_filtered['vote_average'], bins=20, kde=True, color='blue', edgecolor='black')
ax.set_title("Distribución de Calificaciones")
st.pyplot(fig)

# Comparación de presupuesto y recaudación
st.subheader("Relación Presupuesto vs. Recaudación")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=df_filtered['budget'], y=df_filtered['revenue'], alpha=0.7)
ax.set_xlabel("Presupuesto")
ax.set_ylabel("Recaudación")
ax.set_title("Comparación de Presupuesto y Recaudación")
st.pyplot(fig)

# Mostrar el schema final del dataset
st.subheader("Schema Final del Dataset Procesado")
st.text(", ".join(df.columns))