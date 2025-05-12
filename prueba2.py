# streamlit_app.py
import io, ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix)

st.set_page_config(page_title="Éxito de Películas TMDB", layout="wide")
st.title("🎬 Predicción de Éxito de Películas (TMDB)")

# ------------------------------------------------------------------
# 1. CARGA AUTOMÁTICA DEL CSV (sin file_uploader)
# ------------------------------------------------------------------
from pathlib import Path
CSV_PATH = Path(__file__).parent / "data" / "TMDB_movie_dataset_v11.csv"   # ajusta si lo mueves

@st.cache_data
def load_and_clean_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # --- columnas a descartar ---
    df.drop(columns=[c for c in ['homepage','backdrop_path','poster_path','tagline','imdb_id','original_language',
                                 'original_title','overview','production_countries','spoken_languages',
                                 'keywords'] if c in df.columns], inplace=True)

    # --- numéricas ---
    num_fix = ['budget','revenue','runtime','vote_count','popularity']
    df[num_fix] = df[num_fix].apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- fechas ---
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    df['release_year']  = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_dow']   = df['release_date'].dt.dayofweek
    df['is_summer']     = df['release_month'].isin([6,7,8]).astype(int)
    df = df.query("release_year >= 2000")

    # --- filtros necesarios ---
    df = df.query("budget > 1_000 and revenue > 1_000 and 500 <= vote_count <= 30000")
    df = df[df['status'] == 'Released'].copy()
    df['adult'] = df['adult'].astype(str).str.lower().isin(['true','1','yes']).astype(int)

    # --- parsear JSON-like ---
    def parse_col(col, key):
        names, lengths, primary = [], [], []
        for v in col:
            try:
                items = ast.literal_eval(v) if isinstance(v, str) else []
                if isinstance(items, dict): items = [items]
            except Exception:
                items = []
            lst = [d.get(key) for d in items if isinstance(d, dict) and d.get(key)]
            names.append(lst)
            lengths.append(len(lst))
            primary.append(lst[0] if lst else "Unknown")
        return names, lengths, primary

    df['genres_list'], df['n_genres'], df['major_genre'] = parse_col(df['genres'], 'name')
    df['comp_list'],   df['n_comps'],  df['major_comp']  = parse_col(df['production_companies'], 'name')
    df.drop(columns=['genres','production_companies','status'], inplace=True)

    # --- features extras ---
    df['roi'] = (df['revenue'] / df['budget']).replace([np.inf, -np.inf], 0)

    return df

# Intento de carga automática
if CSV_PATH.exists():
    df = load_and_clean_data(CSV_PATH)
else:
    st.error(f"No se encontró el archivo CSV en: {CSV_PATH}")
    st.stop()

# ------------------------------------------------------------------
# 2. EDA
# ------------------------------------------------------------------
st.header("📊 Análisis Exploratorio")
st.dataframe(df.head(), use_container_width=True)

num_for_corr = ['budget','revenue','runtime','vote_count','popularity','roi']
fig, ax = plt.subplots(figsize=(7,5))
sns.heatmap(df[num_for_corr].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# ------------------------------------------------------------------
# 3. INGENIERÍA DE FEATURES
# ------------------------------------------------------------------
st.header("🛠️ Feature Engineering")

# Normalizar vote_count y popularity a 0-10 (petición del usuario)
scaler_01 = MinMaxScaler(feature_range=(0,10))
df[['vote_count','popularity']] = scaler_01.fit_transform(df[['vote_count','popularity']])

top_genres = df['major_genre'].value_counts().nlargest(5).index
for g in top_genres:
    df[f'genre_{g}'] = (df['major_genre'] == g).astype(int)

df['is_success'] = (df['revenue'] >= df['budget']).astype(int)

base_feats = ['budget','runtime','vote_count','popularity','release_year',
              'n_genres','n_comps','adult','roi','is_summer']
feature_cols = base_feats + [f'genre_{g}' for g in top_genres] + ['major_comp']
target_col   = 'is_success'

df_model = df.dropna(subset=feature_cols + [target_col]).copy()
num_cols = [c for c in feature_cols if df_model[c].dtype != 'object']
cat_cols = ['major_comp']            # sólo una categórica por ahora

st.write("Numéricas 👉", num_cols)
st.write("Categóricas 👉", cat_cols)

preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore',
                              sparse_output=False,
                              dtype=np.int8), cat_cols),
    ]
)

X = df_model[num_cols + cat_cols]
y = df_model[target_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 4. MODELADO + GRIDSEARCH (cacheado)
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
            {'clf__n_estimators':[100,200],
             'clf__max_depth':[5,10,None]}
        ),
        "HistGB": (
            HistGradientBoostingClassifier(random_state=42),
            {'clf__learning_rate':[0.03,0.1],
             'clf__max_iter':[200],
             'clf__early_stopping':[False]}
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
# 5. EVALUACIÓN + ENSAMBLE
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

# Soft-voting ensemble
voting = VotingClassifier(estimators=list(models.items()), voting='soft')
voting.fit(X_train, y_train)
results["Ensemble"] = evaluate(voting, X_test, y_test)

# ------------------------------------------------------------------
# 6. VISUALIZACIÓN
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

# Feature importance RF
rf = models["RandomForest"].named_steps['clf']
cat_names = (models["RandomForest"]
             .named_steps['prep']
             .named_transformers_['cat']
             .get_feature_names_out(cat_cols))
feat_names = num_cols + list(cat_names)
importances = pd.Series(rf.feature_importances_, index=feat_names).nlargest(10)

fig, ax = plt.subplots(figsize=(6,3))
importances.plot.bar(ax=ax); ax.set_title("Top 10 Features (RF)")
st.pyplot(fig)

# ------------------------------------------------------------------
# 7. PREDICCIÓN INDIVIDUAL
# ------------------------------------------------------------------
st.header("🔮 Predicción individual")
with st.form("form_pred"):
    user_in = {}
    for f in base_feats:
        user_in[f] = st.number_input(f, value=float(df_model[f].median()))
    user_in['major_comp'] = st.selectbox("Compañía principal", df_model['major_comp'].value_counts().index[:20])
    for g in top_genres:
        user_in[f'genre_{g}'] = 1 if g == df_model['major_genre'].mode()[0] else 0
    if st.form_submit_button("Predecir"):
        x_new = pd.DataFrame([user_in])
        prob  = voting.predict_proba(x_new)[0,1]
        st.success(f"Probabilidad de éxito: **{prob:.2%}**")

st.caption("© ITESO – Minería de Datos 2025")

