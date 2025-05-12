import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, roc_curve,
                             precision_recall_curve, auc as calc_auc, confusion_matrix)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from fpdf import FPDF
import ast
import io

# --- Configuración de la Página de Streamlit ---
st.set_page_config(layout="wide", page_title="Análisis y Modelado de Éxito de Películas TMDB")

# --- Título Principal de la Aplicación ---
st.title("🎬 Predicción de Éxito de Películas (TMDB)")
st.markdown("""
Esta aplicación realiza análisis exploratorio de datos, preprocesamiento, búsqueda de hiperparámetros,
modelado predictivo y generación de reporte para clasificar películas como éxito o fracaso.
Incluye validación cruzada, GridSearch, ensemble, curvas ROC/PR, y predicción individual.
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
    drop_cols = ['homepage','backdrop_path','poster_path','tagline','imdb_id', 'original_language', 'original_title','overview','production_countries','spoken_languages','keywords']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Conversión a numérico y relleno de nulos
    for c in ['budget','revenue','runtime','vote_count','popularity']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Fecha de estreno y extracción de componentes
    if 'release_date' in df.columns:
        df.dropna(subset=['release_date'], inplace=True)
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df.dropna(subset=['release_date'], inplace=True)
        df['release_year'] = df['release_date'].dt.year
        df['release_month'] = df['release_date'].dt.month
        df['release_day_of_week'] = df['release_date'].dt.dayofweek
        df['is_summer_release'] = df['release_month'].isin([6,7,8]).astype(int)
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
        df['adult'] = df['adult'].apply(lambda x: 1 if str(x).lower() in ['true','1','yes'] else 0)
    else:
        df['adult'] = 0

    # Procesar columnas JSON-like (genres, production_companies)
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
        df['genre_list'], df['num_genres'], df['primary_genre'] = parse_column(df['genres'], 'name')
        df.drop(columns=['genres'], inplace=True)
    else:
        df['genre_list'], df['num_genres'], df['primary_genre'] = [[]]*len(df), [0]*len(df), ['Unknown']*len(df)

    if 'production_companies' in df.columns:
        df['company_list'], df['num_companies'], df['primary_company'] = parse_column(df['production_companies'], 'name')
        df.drop(columns=['production_companies'], inplace=True)
    else:
        df['company_list'], df['num_companies'], df['primary_company'] = [[]]*len(df), [0]*len(df), ['Unknown']*len(df)

    # ROI como feature adicional
    df['roi'] = (df['revenue'] / df['budget']).replace([np.inf, -np.inf], 0)

    return df

# --- Cargar datos ---
df = load_and_clean_data('data/TMDB_movie_dataset_v11.csv')
if df.empty:
    st.stop()

# Definir variable objetivo de clasificación
df['is_success'] = (df['revenue'] >= df['budget']).astype(int)

# ------------------------------------------
# Sección 1: Análisis Exploratorio (EDA)
# ------------------------------------------
st.header("📊 Análisis Exploratorio de Datos")
# Vista previa y estadísticas básicas
st.subheader("Vista Previa del Dataset")
st.dataframe(df.head(10))

st.subheader("Estadísticas Descriptivas")
st.dataframe(df[['budget','revenue','runtime','vote_count','popularity','roi']].describe())

# Correlaciones
numeric_feats = ['budget','revenue','runtime','vote_count','popularity','roi']
corr = df[numeric_feats].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# ------------------------------------------
# Sección 2: Ingeniería de Features y Preprocesamiento
# ------------------------------------------
st.header("🛠️ Ingeniería de Features & Preprocesamiento")
# Definir features base y categoría top géneros
base_feats = ['budget','runtime','vote_count','popularity','release_year',
              'num_genres','num_companies','adult','roi','is_summer_release']
top_genres = df['primary_genre'].value_counts().nlargest(5).index.tolist()
for g in top_genres:
    df[f'genre_{g}'] = (df['primary_genre']==g).astype(int)

features = base_feats + [f'genre_{g}' for g in top_genres] + ['primary_company']
target = 'is_success'

df_model = df.dropna(subset=features+[target])

# Separar numéricas y categóricas
num_cols = [c for c in features if np.issubdtype(df_model[c].dtype, np.number)]
cat_cols = ['primary_company']

st.markdown(f"**Numéricas:** {num_cols}")
st.markdown(f"**Categóricas:** {cat_cols}")

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
], remainder='drop')

# División train/test
X = df_model[num_cols+cat_cols]
y = df_model[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------
# Sección 3: Búsqueda de Hiperparámetros y Validación Cruzada
# ------------------------------------------
st.header("🔎 GridSearch & Validación Cruzada")
models = {
    'Regresión Logística': (LogisticRegression(solver='liblinear', max_iter=1000), {'clf__C':[0.01,0.1,1,10]}),
    'Random Forest': (RandomForestClassifier(random_state=42), {'clf__n_estimators':[50,100],'clf__max_depth':[5,10,None]}),
    'HistGB': (HistGradientBoostingClassifier(random_state=42), {'clf__learning_rate':[0.01,0.1],'clf__max_iter':[100,200]})
}
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, (mdl, params) in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', mdl)])
    gs = GridSearchCV(pipe, params, cv=cv, scoring='roc_auc')
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:,1]
    results[name] = {
        'model': best,
        'accuracy': accuracy_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba),
        'fpr,tpr,_': roc_curve(y_test, y_proba),
        'prec,rec,_': precision_recall_curve(y_test, y_proba)
    }
    st.write(f"{name} mejor C{gs.best_params_}: AUC={results[name]['auc']:.3f} Acc={results[name]['accuracy']:.3f}")

# ------------------------------------------
# Ensamble soft voting de pipelines
# ------------------------------------------
# Usamos los pipelines ya optimizados (cada uno contiene preprocesador + clasificador)
ens_estimators = [(name, res['model']) for name, res in results.items()]
ens = VotingClassifier(estimators=ens_estimators, voting='soft')
ens.fit(X_train, y_train)
# Evaluación del ensemble
y_pred_e = ens.predict(X_test)
y_proba_e = ens.predict_proba(X_test)[:, 1]
results['Ensamble'] = {
    'model': ens,
    'accuracy': accuracy_score(y_test, y_pred_e),
    'auc': roc_auc_score(y_test, y_proba_e),
    'fpr,tpr,_': roc_curve(y_test, y_proba_e),
    'prec,rec,_': precision_recall_curve(y_test, y_proba_e)
}
st.write(f"Ensamble: AUC={results['Ensamble']['auc']:.3f} Acc={results['Ensamble']['accuracy']:.3f}")

# Umbral en sidebar
thresh = st.sidebar.slider('Umbral probabilidad', 0.0, 1.0, 0.5)

# ------------------------------------------
# Sección 4: Resultados y Visualizaciones
# ------------------------------------------
st.header('📈 Resultados de Modelos')
for name, res in results.items():
    st.subheader(name)
    # Matriz de confusión
    y_thresh = (res['model'].predict_proba(X_test)[:,1] >= thresh).astype(int)
    cm = confusion_matrix(y_test, y_thresh)
    fig, ax = plt.subplots(); sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_title(f'Matriz confusión {name}'); st.pyplot(fig)
    # Curva ROC
    fpr, tpr, _ = res['fpr,tpr,_']
    fig, ax = plt.subplots(); ax.plot(fpr, tpr); ax.set_title(f'ROC {name}'); st.pyplot(fig)
    # Curva PR
    prec, rec, _ = res['prec,rec,_']
    fig, ax = plt.subplots(); ax.plot(rec, prec); ax.set_title(f'PR {name}'); st.pyplot(fig)

# Importancia features Random Forest
st.header('🧩 Importancia de Features (Random Forest)')
rf = results['Random Forest']['model'].named_steps['clf']
feat_names = num_cols + list(results['Random Forest']['model'].named_steps['prep'].named_transformers_['cat'].get_feature_names_out(cat_cols))
imp = pd.Series(rf.feature_importances_, index=feat_names).nlargest(10)
fig, ax = plt.subplots(); imp.plot.bar(ax=ax); ax.set_title('Top 10 Features'); st.pyplot(fig)

# ------------------------------------------
# Sección 5: Predicción Individual
# ------------------------------------------
st.header('🔮 Predicción Individual')
with st.form('form_pred'):
    inputs = {}
    for c in base_feats:
        inputs[c] = st.number_input(c, value=float(df[c].median()))
    inputs['primary_company'] = st.selectbox('primary_company', sorted(df['primary_company'].unique())[:20])
    inputs['primary_genre']   = st.selectbox('primary_genre', top_genres)
    for g in top_genres:
        inputs[f'genre_{g}'] = 1 if inputs['primary_genre']==g else 0
    submitted = st.form_submit_button('Predecir')
    if submitted:
        xnew = pd.DataFrame([inputs])
        ensemble_model = results['Ensamble']['model']
        prob = ensemble_model.predict_proba(xnew)[0,1]
        st.write(f'Probabilidad éxito: {prob:.2f}')
        st.write('Resultado:', 'Éxito' if prob>=thresh else 'Fracaso')

# ------------------------------------------
# Sección 6: Reporte PDF
# ------------------------------------------
def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Reporte de Modelos TMDB', ln=1)
    pdf.set_font('Arial', '', 12)
    for name, res in results.items():
        pdf.cell(0, 8, f"{name}: Acc={res['accuracy']:.3f}, AUC={res['auc']:.3f}", ln=1)
    # Generar bytes del PDF
    pdf_output = pdf.output(dest='S').encode('latin-1')
    buf = io.BytesIO(pdf_output)
    return buf

pdf_buf = create_pdf()
st.download_button('📥 Descargar Reporte PDF', data=pdf_buf, file_name='reporte_tmdb.pdf', mime='application/pdf')

st.markdown('---')
st.write('© ITESO - Proyecto Final - Minería de Datos')
