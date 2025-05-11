import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import ast

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis y Modelado de Éxito de Películas TMDB")

# --- Título Principal de la Aplicación ---
st.title("🎬 Predicción de Éxito de Películas (TMDB)")
st.markdown("""
Esta aplicación realiza análisis exploratorio de datos, preprocesamiento y modelado predictivo para clasificar
películas como éxito o fracaso basándose en presupuesto, fecha de estreno, género y otras características.
Cumple los requisitos de la segunda entrega del proyecto.
""")

# --- Carga y Limpieza Inicial de Datos ---
@st.cache_data
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"Archivo no encontrado: {file_path}")
        return pd.DataFrame()

    # Columnas irrelevantes
    drop_cols = ['homepage', 'backdrop_path', 'poster_path', 'tagline', 'imdb_id',
                 'original_title', 'overview', 'production_countries', 'spoken_languages', 'keywords']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Conversión a numérico y relleno de nulos
    for col in ['budget', 'revenue', 'runtime', 'vote_count', 'popularity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Fecha de estreno y extracción de componentes
    if 'release_date' in df.columns:
        df = df.dropna(subset=['release_date'])
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.dropna(subset=['release_date'])
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day_of_week'] = df['release_date'].dt.dayofweek
        df = df[df['release_year'] >= 2000]

    # Filtros básicos de presupuesto e ingresos
    df = df[(df['budget'] > 1000) & (df['revenue'] > 1000)]

    # Solo películas released
    if 'status' in df.columns:
        df = df[df['status'] == 'Released']
        df.drop(columns=['status'], inplace=True)

    # Filtrar vote_count
    df = df[(df['vote_count'] >= 500) & (df['vote_count'] <= 30000)]

    # Variable adult como binaria
    if 'adult' in df.columns:
        df['adult'] = df['adult'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0)
    else:
        df['adult'] = 0

    # Procesar columnas JSON-like
    def parse_column(series, key_name):
        names, counts, primary = [], [], []
        for v in series:
            try:
                items = ast.literal_eval(v) if isinstance(v, str) else []
                if isinstance(items, dict):
                    items = [items]
            except Exception:
                items = []
            name_list = [it.get(key_name) for it in items if isinstance(it, dict) and it.get(key_name)]
            names.append(name_list)
            counts.append(len(name_list))
            primary.append(name_list[0] if name_list else 'Unknown')
        return names, counts, primary

    if 'genres' in df.columns:
        df['genre_names_list'], df['num_genres'], df['primary_genre'] = parse_column(df['genres'], 'name')
        df.drop(columns=['genres'], inplace=True)
    else:
        df['genre_names_list'] = [[] for _ in range(len(df))]
        df['num_genres'] = 0
        df['primary_genre'] = 'Unknown'

    if 'production_companies' in df.columns:
        df['company_names_list'], df['num_companies'], df['primary_company'] = parse_column(df['production_companies'], 'name')
        df.drop(columns=['production_companies'], inplace=True)
    else:
        df['company_names_list'] = [[] for _ in range(len(df))]
        df['num_companies'] = 0
        df['primary_company'] = 'Unknown'

    return df

# Cargar datos
df = load_and_clean_data('data/TMDB_movie_dataset_v11.csv')
if df.empty:
    st.stop()

# Definir variable objetivo: éxito si revenue >= budget
df['is_success'] = (df['revenue'] >= df['budget']).astype(int)

# ------------------------------------------
# Sección 1: Análisis Exploratorio
# ------------------------------------------
st.header("📊 Exploración de Datos")
st.write(df.head())

# ------------------------------------------
# Sección 2: Preprocesamiento y Pipeline
# ------------------------------------------
st.header("🛠️ Preparación y Modelado")

# Definir características y objetivo para modelado
features = [
    'budget', 'runtime', 'vote_count', 'popularity', 'release_year',
    'num_genres', 'num_companies', 'adult',
    'primary_genre', 'primary_company'
]
target = 'is_success'

# Filtrar filas con datos completos
df_model = df.dropna(subset=features + [target])

# Identificar columnas numéricas y categóricas
numerical_cols = [col for col in features if col in df_model.columns and np.issubdtype(df_model[col].dtype, np.number)]
categorical_cols = [col for col in features if col in df_model.columns and col not in numerical_cols]

# Mostrar selección de columnas
st.markdown(f"**Columnas numéricas:** {', '.join(numerical_cols)}")
st.markdown(f"**Columnas categóricas:** {', '.join(categorical_cols)}")

# Configurar preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='drop'
)

# Dividir datos en train/test
target_series = df_model[target]
features_df = df_model[numerical_cols + categorical_cols]
X_train, X_test, y_train, y_test = train_test_split(
    features_df, target_series, test_size=0.2, random_state=42, stratify=target_series
)

# ------------------------------------------
# Sección 3: Modelos
# ------------------------------------------
st.subheader("Modelos de Clasificación de Éxito")
models = {
    'Regresión Logística': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    st.markdown(f"**{name}:** Accuracy={acc:.3f}, AUC={auc:.3f}")
    st.text(classification_report(y_test, y_pred, zero_division=0))

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_title(f"ROC Curve - {name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    st.pyplot(fig)

# ------------------------------------------
# Fin de la Aplicación
# ------------------------------------------
st.markdown("---")
st.write("© ITESO - Proyecto Final de Minería de Datos")
