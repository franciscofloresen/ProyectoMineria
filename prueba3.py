# streamlit_app.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

st.set_page_config(page_title="Éxito de Películas TMDB", layout="wide")
st.title("🎬 Predicción de Éxito de Películas (TMDB)")

# ------------------------------------------------------------------
# 1. CARGAR EL DATASET ORIGINAL
# ------------------------------------------------------------------
st.info("Cargando el dataset original...")
df = pd.read_csv('data/TMDB_movie_dataset_v11.csv', low_memory=False)

# 2. Eliminar columnas irrelevantes
cols_to_drop = [
    'homepage', 'backdrop_path', 'poster_path', 'tagline', 'imdb_id',
    'original_language', 'original_title', 'overview', 'production_countries',
    'spoken_languages', 'keywords'
]
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 3. Manejo de valores nulos
num_fix = ['budget', 'revenue', 'runtime', 'vote_count', 'popularity']
df[num_fix] = df[num_fix].apply(pd.to_numeric, errors='coerce').fillna(0)

# 4. Extraer año, mes y día de la semana de 'release_date'
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df = df.dropna(subset=['release_date'])
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_dow'] = df['release_date'].dt.dayofweek
df['is_summer'] = df['release_month'].isin([6, 7, 8]).astype(int)

# 5. Filtrar películas desde el año 2000 y con votos entre 500 y 30000
df = df[(df['release_year'] >= 2000) & (df['vote_count'] >= 500) & (df['vote_count'] <= 30000)]

# 6. Filtrar películas con status 'Released'
df = df[df['status'] == 'Released']
df.drop(columns=['status'], inplace=True)

# 7. Convertir la columna 'adult' a binario
df['adult'] = df['adult'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)

# 8. Calcular el ROI (Retorno de Inversión)
df['roi'] = (df['revenue'] / df['budget']).replace([np.inf, -np.inf], 0)

# 9. Parsear columnas con múltiples valores separados por comas
def parse_col(col, key=None):
    """
    Parsea columnas que contienen valores separados por comas.
   
    Args:
        col: Columna del DataFrame con strings de valores separados por comas
        key: Parámetro ignorado, mantenido para compatibilidad con la función original
             (no se usa en esta versión ya que procesamos CSV, no JSON)
   
    Returns:
        Lista con el primer valor de cada celda (o "Unknown" si está vacía)
    """
    primary = []
   
    for v in col:
        # Manejo de valores nulos o vacíos
        if v is None or pd.isna(v) or (isinstance(v, str) and v.strip() == ''):
            primary.append("Unknown")
            continue
           
        try:
            # Si el valor ya es un iterable (lista, etc.) pero no string
            if hasattr(v, '__iter__') and not isinstance(v, str):
                values = list(v)
            else:
                # Dividir el string por comas y eliminar espacios en blanco
                values = [item.strip() for item in str(v).split(',')]
               
            # Filtrar valores vacíos
            values = [item for item in values if item]
               
            # Tomar el primer valor o usar "Unknown" si la lista está vacía
            primary.append(values[0] if values else "Unknown")
           
        except Exception:
            # En caso de error, usar "Unknown"
            primary.append("Unknown")
           
    return primary
 
# Ejemplo de uso:
df['major_genre'] = parse_col(df['genres'])
df['major_comp'] = parse_col(df['production_companies'])
df.drop(columns=['genres', 'production_companies'], inplace=True)


# 10. Crear la columna de éxito (1 si revenue >= budget)
df['is_success'] = (df['revenue'] >= df['budget']).astype(int)

# 11. Normalizar columnas 'vote_count' y 'popularity'
scaler_01 = MinMaxScaler(feature_range=(0, 10))
df[['vote_count', 'popularity']] = scaler_01.fit_transform(df[['vote_count', 'popularity']])

# 12. Guardar el dataset procesado
df.to_csv('data/peliculas_procesadas.csv', index=False)
st.success("Archivo procesado guardado como peliculas_procesadas.csv")

# ------------------------------------------------------------------
# 2. CARGAR EL ARCHIVO PROCESADO
# ------------------------------------------------------------------
st.info("Cargando el dataset procesado...")
df = pd.read_csv('data/peliculas_procesadas.csv', low_memory=False)

# ------------------------------------------------------------------
# 3. ANÁLISIS EXPLORATORIO
# ------------------------------------------------------------------
st.header("📊 Análisis Exploratorio")
st.dataframe(df.head(), use_container_width=True)

num_for_corr = ['budget', 'revenue', 'runtime', 'vote_count', 'popularity', 'roi']
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(df[num_for_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# ------------------------------------------------------------------
# 4. INGENIERÍA DE FEATURES
# ------------------------------------------------------------------
st.header("🛠️ Feature Engineering")

# Verificar si la columna 'major_genre' existe antes de calcular los géneros más frecuentes
if 'major_genre' in df.columns:
    top_genres = df['major_genre'].value_counts().nlargest(5).index
else:
    top_genres = []

# Crear columnas binarias para los géneros más comunes (opcional si necesitas alguna binaria específica)
for g in top_genres:
    col_name = f'genre_{g}'
    if col_name not in df.columns:
        df[col_name] = 0  # Asegurar que siempre existan las columnas binarias
    else:
        df[col_name] = (df['major_genre'] == g).astype(int)

# Calcular el número de géneros
df['n_genres'] = df['major_genre'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# Calcular el número de compañías
df['n_comps'] = df['major_comp'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)

# Definir las características base
base_feats = ['budget', 'runtime', 'vote_count', 'popularity', 'release_year', 
              'n_genres', 'n_comps', 'adult', 'roi', 'is_summer']

# Generar la lista de características finales de manera segura
feature_cols = base_feats + ['major_genre', 'major_comp']
target_col = 'is_success'

# Verificar que las columnas realmente existan en el DataFrame
feature_cols = [col for col in feature_cols if col in df.columns]

# Crear el DataFrame final para el modelado
df_model = df.dropna(subset=feature_cols + [target_col]).copy()
num_cols = [c for c in base_feats if df_model[c].dtype != 'object']

# Definir columnas categóricas (solo 'major_genre' y 'major_comp')
cat_cols = ['major_genre', 'major_comp'] if 'major_comp' in df_model.columns else ['major_genre']

st.write("Numéricas 👉", num_cols)
st.write("Categóricas 👉", cat_cols)

# Aplicar LabelEncoder a las columnas categóricas
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_model[f'{col}_encoded'] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Columnas para el modelo después de la codificación
encoded_cat_cols = [f'{col}_encoded' for col in cat_cols]
model_features = num_cols + encoded_cat_cols

# Preprocesamiento con ColumnTransformer solo para las numéricas
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols + encoded_cat_cols),
    ],
    remainder='passthrough'
)

X = df_model[model_features]
y = df_model[target_col]

# Separación en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 5. MODELADO + GRIDSEARCH (cacheado)
# ------------------------------------------------------------------
@st.cache_resource(hash_funcs={'GridSearchCV': id})
def train_models(X_tr, y_tr):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic": (
            LogisticRegression(solver='liblinear', max_iter=1000),
            {'clf__C':[0.01,0.1,1,10]}
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {'clf__n_estimators':[100, 200],
             'clf__max_depth':[None, 10, 20],
             'clf__min_samples_split':[2, 5]}
        )
    }

    res = {}
    for name, (est, grid) in models.items():
        pipe = Pipeline([('prep', preprocess), ('clf', est)])
        gs   = GridSearchCV(pipe, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        gs.fit(X_tr, y_tr)
        res[name] = gs.best_estimator_
    return res

models = train_models(X_train, y_train)

# ------------------------------------------------------------------
# 6. EVALUACIÓN DE MODELOS
# ------------------------------------------------------------------
def evaluate(model, X_te, y_te):
    y_prob = model.predict_proba(X_te)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)
    
    return {
        "acc": accuracy_score(y_te, y_pred),
        "auc": roc_auc_score(y_te, y_prob),
        "roc": roc_curve(y_te, y_prob),
        "pr":  precision_recall_curve(y_te, y_prob),
        "cm":  confusion_matrix(y_te, y_pred)
    }

results = {name: evaluate(m, X_test, y_test) for name, m in models.items()}

# ------------------------------------------------------------------
# 7. VISUALIZACIÓN
# ------------------------------------------------------------------
st.header("📈 Resultados")
for name, res in results.items():
    st.subheader(name)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Accuracy", f"{res['acc']:.3f}")
    with col2: st.metric("ROC-AUC",  f"{res['auc']:.3f}")

    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(res['cm'], annot=True, fmt='d', cbar=False, ax=ax)
    ax.set_title("Confusión"); st.pyplot(fig)

    # ROC
    fpr, tpr, _ = res['roc']
    fig, ax = plt.subplots(); ax.plot(fpr, tpr); ax.set_title("ROC"); st.pyplot(fig)

    # Importancia de características (solo para RandomForest)
    if name == "RandomForest":
        try:
            # Obtener el modelo RandomForest subyacente
            rf_model = models["RandomForest"].named_steps['clf']
            
            # Obtener importancia de características
            importances = rf_model.feature_importances_
            
            # Crear un DataFrame para visualizar la importancia
            feature_importance_df = pd.DataFrame({
                'Feature': model_features,
                'Importance': importances
            })
            
            # Asignar nombres originales a las características codificadas
            for col in cat_cols:
                encoded_col = f'{col}_encoded'
                if encoded_col in feature_importance_df['Feature'].values:
                    idx = feature_importance_df[feature_importance_df['Feature'] == encoded_col].index[0]
                    feature_importance_df.loc[idx, 'Feature'] = col
            
            # Ordenar por importancia descendente
            feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
            
            # Visualizar las 15 características más importantes
            top_features = feature_importance_df.head(15)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(range(len(top_features)), top_features['Importance'], align='center')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_title("Importancia de características")
            ax.set_xlabel("Importancia")
            
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"No se pudo mostrar la importancia de características: {e}")

# ------------------------------------------------------------------
# 8. PREDICCIÓN INDIVIDUAL
# ------------------------------------------------------------------
st.header("🔮 Predicción individual")

# Seleccionar el mejor modelo basado en AUC
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_model = models[best_model_name]
st.info(f"📈 Utilizando el mejor modelo: **{best_model_name}** (AUC: {results[best_model_name]['auc']:.3f})")

with st.form("form_pred"):
    user_in = {}
    for f in base_feats:
        if f in df_model.columns:
            user_in[f] = st.number_input(f, value=float(df_model[f].median()))
    
    # Usar selectbox para las categóricas originales, no las codificadas
    for col in cat_cols:
        user_in[col] = st.selectbox(f"{col.replace('_', ' ').title()}", 
                                   df_model[col].value_counts().index[:20])

    # Botón de envío
    submitted = st.form_submit_button("Predecir")
    if submitted:
        try:
            # Crear DataFrame con la entrada del usuario
            x_new = pd.DataFrame([user_in])
            
            # Codificar las categóricas usando los LabelEncoder existentes
            for col in cat_cols:
                le = label_encoders[col]
                # Manejar valores desconocidos
                if x_new[col].iloc[0] in le.classes_:
                    x_new[f'{col}_encoded'] = le.transform(x_new[col])
                else:
                    # Usar el valor más común si el valor es desconocido
                    st.warning(f"Valor '{x_new[col].iloc[0]}' no visto en entrenamiento para '{col}'. Usando un valor común.")
                    x_new[f'{col}_encoded'] = 0
            
            # Seleccionar solo las columnas que el modelo espera
            x_pred = x_new[model_features]
            
            # Realizar la predicción
            prob = best_model.predict_proba(x_pred)[0, 1]
            st.success(f"Probabilidad de éxito: **{prob:.2%}**")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")

st.caption("© ITESO – Minería de Datos 2025")