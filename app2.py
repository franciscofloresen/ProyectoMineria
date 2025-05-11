import streamlit as st
import pandas as pd
import numpy as np
import ast  # Para convertir strings de listas/diccionarios a objetos Python
import json  # Para manejar estructuras JSON complejas
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Proyecto Minería de Datos: Películas TMDB")

# --- RUTA DEL DATASET ---
DATASET_PATH = "data/TMDB_movie_dataset_v11.csv"


# --- Funciones Auxiliares (de la primera entrega) ---

def load_data(file_path):
    """Carga los datos desde una ruta de archivo CSV."""
    try:
        df = pd.read_csv(file_path)
        # Intentar inferir tipos de datos problemáticos
        for col in ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        st.success(f"Dataset cargado exitosamente desde: {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo en la ruta especificada: {file_path}")
        st.error(
            f"Asegúrate de que el archivo {DATASET_PATH} esté en la misma carpeta que el script, o actualiza la variable DATASET_PATH.")
        return None
    except Exception as e:
        st.error(f"Error al cargar el archivo {file_path}: {e}")
        return None


def display_dataframe_info(df, title="Información del DataFrame"):
    """Muestra información general del DataFrame."""
    st.subheader(title)
    st.write("Primeras 5 filas:")
    st.dataframe(df.head())

    st.write("Información general (Tipos de datos y Nulos):")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.write("Estadísticas Descriptivas:")
    st.dataframe(df.describe(include='all').transpose())


def display_missing_values(df, title="Valores Faltantes"):
    """Muestra un resumen de valores faltantes."""
    st.subheader(title)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.write("Conteo de valores faltantes por columna:")
        st.dataframe(missing_data.sort_values(ascending=False))
        # Gráfico de barras para visualizar nulos
        fig, ax = plt.subplots(figsize=(10, max(5, len(missing_data) * 0.3)))
        missing_data.sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_title("Valores Faltantes por Columna")
        ax.set_ylabel("Cantidad de Nulos")
        st.pyplot(fig)

    else:
        st.success("¡No hay valores faltantes significativos en el dataset para las columnas procesadas!")


def safe_literal_eval(val):
    """Intenta convertir un string a un objeto Python (lista/diccionario)."""
    if pd.isna(val):
        return []
    if not isinstance(val, str):  # Si ya es una lista o dict (poco probable desde CSV crudo)
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        try:  # Intento de parsear como JSON si ast falla (maneja comillas dobles)
            return json.loads(val.replace("'", "\""))  # Reemplazo simple, puede no ser robusto
        except json.JSONDecodeError:
            # st.warning(f"No se pudo parsear: {val}. Se devuelve lista vacía.")
            return []
    except Exception:
        return []


def extract_names_from_json_like_list(series, key_name='name'):
    """Extrae una lista de nombres de una columna que contiene strings de listas de diccionarios."""
    all_items_list = []
    for item_str in series:
        items = []
        parsed_list = safe_literal_eval(item_str)
        if isinstance(parsed_list, list):
            for entry in parsed_list:
                if isinstance(entry, dict) and key_name in entry:
                    items.append(entry[key_name])
        all_items_list.append(items)
    return pd.Series(all_items_list, index=series.index)


# --- Estado de la Aplicación (Session State) ---
if 'df_processed' not in st.session_state:
    st.session_state.df_processed = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.preprocessor = None
    st.session_state.selected_features = []

# --- Título de la Aplicación ---
st.title("🎬 Proyecto Minería de Datos: Análisis de Películas TMDB")

# --- Carga de Datos ---
st.sidebar.title("Navegación")
st.sidebar.header("Carga Inicial")
st.sidebar.info(f"Intentando cargar dataset desde: {DATASET_PATH}")

df_original = load_data(DATASET_PATH)

if df_original is None:
    st.error("El DataFrame no pudo ser cargado. Por favor, revisa la ruta del archivo y los mensajes de error.")
    st.stop()

# Solo procesar si no se ha hecho antes o si se reinicia
if st.session_state.df_processed is None:
    df = df_original.copy()
    # --- Procesamiento de la Primera Entrega (se ejecuta una vez y se guarda en session_state) ---
    # (Este bloque se podría mover a una función para limpieza)
    df_processed = df.copy()

    # 3.1. Limpieza y Manejo de Valores Nulos
    for col in ['budget', 'revenue', 'runtime']:  # Ya convertidos en load_data
        if df_processed[col].isnull().any():
            if col in ['budget', 'revenue']:  # Para budget y revenue, 0 podría ser un valor válido o faltante
                # Considerar estrategias más avanzadas si el tiempo lo permite (ej. imputar por género)
                # Por ahora, si es para regresión de revenue, filas con revenue nulo o cero podrían ser problemáticas
                df_processed[col].fillna(0, inplace=True)  # Llenar nulos con 0 por ahora
            elif col == 'runtime':
                median_runtime = df_processed[col].median()
                if pd.notna(median_runtime):
                    df_processed[col].fillna(median_runtime, inplace=True)

    rows_before_rd_cleaning = len(df_processed)
    df_processed.dropna(subset=['release_date'], inplace=True)
    df_processed = df_processed[df_processed['release_date'].astype(str).str.strip() != '']
    df_processed['release_date'] = pd.to_datetime(df_processed['release_date'], errors='coerce')
    df_processed.dropna(subset=['release_date'], inplace=True)

    text_cols_to_fill = ['overview', 'tagline', 'title', 'original_title']
    for col in text_cols_to_fill:
        if col in df_processed.columns:
            df_processed[col].fillna("", inplace=True)

    # 3.2. Transformación de Datos
    if 'release_date' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['release_date']):
        df_processed['release_year'] = df_processed['release_date'].dt.year
        df_processed['release_month'] = df_processed['release_date'].dt.month
        df_processed['release_dayofweek'] = df_processed['release_date'].dt.dayofweek

    json_like_cols_data = {
        'genres': 'name',
        'production_companies': 'name',
        'production_countries': 'name',
        'spoken_languages': 'english_name',
        'keywords': 'name'
    }

    temp_cols_created_for_processing = []
    for col_name, key_for_name in json_like_cols_data.items():
        if col_name in df_processed.columns:
            df_processed[col_name].fillna('[]', inplace=True)

            list_col = f'{col_name}_list'
            first_col = f'first_{col_name}'  # Podría ser útil para OHE
            num_col = f'num_{col_name}'

            temp_cols_created_for_processing.extend([list_col, first_col, num_col])

            extracted_names = extract_names_from_json_like_list(df_processed[col_name], key_name=key_for_name)
            df_processed[list_col] = extracted_names

            df_processed[first_col] = df_processed[list_col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
            df_processed[num_col] = df_processed[list_col].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # 3.3. Creación de Nuevas Características
    if 'revenue' in df_processed.columns and 'budget' in df_processed.columns:
        df_processed['profit'] = df_processed['revenue'] - df_processed['budget']
        df_processed['roi'] = np.where(
            (df_processed['budget'] > 0) & pd.notna(df_processed['budget']) & pd.notna(df_processed['profit']),
            (df_processed['profit'] / df_processed['budget']) * 100,
            0
        )

    # 3.4. Selección/Eliminación de Columnas (Ajustado para conservar num_*)
    columns_to_remove_after_processing = [
        'genres_list', 'first_genres',  # num_genres se conserva si se creó
        'production_companies_list', 'first_production_companies',  # num_production_companies se conserva
        'production_countries_list', 'first_production_countries',  # num_production_countries se conserva
        'spoken_languages_list', 'first_spoken_languages',  # num_spoken_languages se conserva
        'keywords_list', 'first_keywords'  # num_keywords se conserva
    ]
    # También eliminar las columnas JSON originales si ya no se necesitan para modelado directo
    # y otras columnas no útiles para modelado general.
    other_cols_to_drop = ['id', 'imdb_id', 'poster_path', 'backdrop_path', 'homepage', 'overview', 'tagline',
                          'original_title',  # Usar 'title'
                          'genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords'
                          # Originales JSON
                          ]

    all_cols_to_drop = list(set(columns_to_remove_after_processing + other_cols_to_drop))
    existing_cols_to_drop = [col for col in all_cols_to_drop if col in df_processed.columns]
    if existing_cols_to_drop:
        df_processed.drop(columns=existing_cols_to_drop, inplace=True, errors='ignore')

    st.session_state.df_processed = df_processed.copy()
    st.sidebar.success("Procesamiento inicial completado.")

# Usar el DataFrame procesado desde el estado de la sesión
df_processed = st.session_state.df_processed

if df_processed is None:  # Fallback si algo salió mal
    st.error("El DataFrame procesado no está disponible. Intentando reprocesar...")
    st.session_state.df_processed = None  # Forzar reprocesamiento
    st.experimental_rerun()

# --- Menú de Etapas ---
st.sidebar.markdown("---")
st.sidebar.header("Etapas del Proyecto")
selected_stage = st.sidebar.radio("Selecciona la etapa a visualizar:",
                                  ["Entendimiento del Negocio (Entrega 1)",
                                   "Entendimiento de los Datos (Entrega 1)",
                                   "Preparación de los Datos (Entrega 1)",
                                   "Análisis de Datos para Modelado (Entrega 2)",
                                   "Preparación de Datos para Modelado (Entrega 2)",
                                   "Modelado y Evaluación (Entrega 2)"])
st.sidebar.markdown("---")

# --- Contenido de las Etapas ---

if selected_stage == "Entendimiento del Negocio (Entrega 1)":
    st.header("1. Entendimiento del Negocio")
    st.markdown("""
    **Problema de Negocio:**
    Las productoras de cine, distribuidoras e inversores buscan maximizar el retorno de su inversión y entender qué factores hacen que una película sea exitosa. El éxito puede medirse en términos de recaudación (taquilla), aceptación por parte de la crítica y el público (calificaciones), o una combinación de ambos. Tomar decisiones informadas sobre qué proyectos financiar, cómo promocionarlos, o qué expectativas tener es crucial en una industria de alto riesgo y alta inversión.

    **Objetivo del Proyecto de Minería de Datos:**
    Desarrollar un modelo predictivo que, utilizando las características disponibles en el dataset de TMDB, pueda:
    1.  Predecir una métrica de éxito continuo (ej. ingresos brutos - `revenue`, o calificación promedio - `vote_average`).
    2.  Clasificar las películas en categorías de éxito (ej. "fracaso", "éxito moderado", "taquillazo") basadas en umbrales definidos.

    Este análisis ayudará a identificar patrones y factores clave que influyen en el desempeño de una película, proporcionando insights valiosos para la toma de decisiones en la industria cinematográfica.
    """)

elif selected_stage == "Entendimiento de los Datos (Entrega 1)":
    st.header("2. Entendimiento de los Datos (Dataset Original)")
    st.markdown(f"Dataset original cargado desde: `{DATASET_PATH}`")
    st.markdown(f"Número de filas: `{df_original.shape[0]}`")
    st.markdown(f"Número de columnas: `{df_original.shape[1]}`")

    display_dataframe_info(df_original, "Información Inicial del Dataset Original")
    display_missing_values(df_original, "Valores Faltantes en Dataset Original")

    st.subheader("Exploración de Columnas Específicas (Dataset Original)")
    column_to_explore = st.selectbox("Selecciona una columna para explorar (Original):", df_original.columns,
                                     key="explore_original")
    if column_to_explore and column_to_explore in df_original.columns:
        if df_original[column_to_explore].nunique() < 30 and df_original[column_to_explore].nunique() > 0:
            st.write(f"Valores únicos en '{column_to_explore}':")
            st.write(df_original[column_to_explore].unique())
        else:
            st.write(f"Algunos ejemplos de valores en '{column_to_explore}' (5 muestras):")
            st.dataframe(df_original[column_to_explore].sample(min(5, len(df_original))))

        json_like_cols_original = ['genres', 'production_companies', 'production_countries', 'spoken_languages',
                                   'keywords']
        if column_to_explore in json_like_cols_original:
            st.write(f"Ejemplo del formato de '{column_to_explore}' (primera fila no nula):")
            first_valid_value = df_original[column_to_explore].dropna().iloc[0] if not df_original[
                column_to_explore].dropna().empty else "No hay valores no nulos"
            st.code(str(first_valid_value))


elif selected_stage == "Preparación de los Datos (Entrega 1)":
    st.header("3. Preparación de los Datos (Resultado de Entrega 1)")
    st.markdown("""
    Esta sección muestra el DataFrame (`df_processed`) después de la limpieza, transformación y creación/eliminación de características 
    realizadas como parte de la primera entrega. Este es el punto de partida para el análisis y modelado de la segunda entrega.
    """)
    if df_processed is not None:
        display_dataframe_info(df_processed, "Dataset Procesado (Resultado Entrega 1)")
        display_missing_values(df_processed, "Valores Faltantes en Dataset Procesado")

        st.download_button(
            label="Descargar Dataset Procesado (CSV)",
            data=df_processed.to_csv(index=False).encode('utf-8'),
            file_name='TMDB_movies_processed_E1.csv',
            mime='text/csv',
            key='download_processed_e1'
        )
    else:
        st.warning("El DataFrame procesado aún no está disponible.")


# --- SEGUNDA ENTREGA ---
elif selected_stage == "Análisis de Datos para Modelado (Entrega 2)":
    st.header("4. Análisis de Datos para Modelado (Segunda Entrega)")
    st.markdown("Partimos del `Dataset Procesado (Resultado Entrega 1)`.")

    if df_processed is not None:
        st.markdown("### Selección de Variable Objetivo y Características")

        # Variable Objetivo
        possible_targets = ['revenue', 'vote_average', 'profit', 'roi']
        target_variable = st.selectbox("Selecciona la Variable Objetivo (Y):",
                                       [t for t in possible_targets if t in df_processed.columns],
                                       index=0, key="target_select")

        st.markdown(f"**Variable Objetivo Seleccionada: `{target_variable}`**")

        # Características Numéricas
        st.markdown("### Análisis de Características Numéricas")
        numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
        if target_variable in numeric_cols:
            numeric_cols.remove(target_variable)  # No incluir el target en las features

        selected_numeric_features = st.multiselect("Selecciona Características Numéricas (X) para Análisis:",
                                                   numeric_cols,
                                                   default=[col for col in
                                                            ['budget', 'popularity', 'runtime', 'vote_count',
                                                             'release_year', 'num_genres', 'num_production_companies']
                                                            if col in numeric_cols],
                                                   key="numeric_features_select")

        if selected_numeric_features and target_variable:
            # Distribución de la Variable Objetivo
            st.markdown(f"#### Distribución de la Variable Objetivo: `{target_variable}`")
            fig_target, ax_target = plt.subplots(figsize=(8, 5))
            sns.histplot(df_processed[target_variable], kde=True, ax=ax_target)
            ax_target.set_title(f"Distribución de {target_variable}")
            ax_target.set_xlabel(target_variable)
            ax_target.set_ylabel("Frecuencia")
            st.pyplot(fig_target)

            # Distribución de Features Numéricas Seleccionadas
            st.markdown("#### Distribuciones de Características Numéricas Seleccionadas")
            for col in selected_numeric_features:
                if col in df_processed.columns:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                    # Histograma
                    sns.histplot(df_processed[col], kde=True, ax=ax[0])
                    ax[0].set_title(f"Distribución de {col}")
                    # Boxplot
                    sns.boxplot(x=df_processed[col], ax=ax[1])
                    ax[1].set_title(f"Boxplot de {col}")
                    st.pyplot(fig)

            # Matriz de Correlación
            st.markdown("#### Matriz de Correlación (entre seleccionadas y objetivo)")
            cols_for_corr = selected_numeric_features + [target_variable]
            corr_matrix = df_processed[cols_for_corr].corr()

            fig_corr, ax_corr = plt.subplots(
                figsize=(max(8, len(cols_for_corr) * 0.8), max(6, len(cols_for_corr) * 0.6)))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax_corr)
            ax_corr.set_title("Mapa de Calor de Correlaciones")
            st.pyplot(fig_corr)

            st.write("Correlación con la variable objetivo:", target_variable)
            st.dataframe(corr_matrix[[target_variable]].sort_values(by=target_variable, ascending=False))

        # Características Categóricas (Análisis Básico)
        st.markdown("### Análisis de Características Categóricas (Ejemplos)")
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        # Columnas como 'first_genre' etc. fueron eliminadas, pero si alguna quedó:
        # O considerar 'original_language', 'status' (si se mantuvo y limpió)

        # Ejemplo con 'original_language' si existe
        if 'original_language' in df_processed.columns:
            st.markdown("#### Distribución de `original_language` (Top 10)")
            if df_processed['original_language'].nunique() > 0:
                fig_lang, ax_lang = plt.subplots(figsize=(10, 6))
                top_langs = df_processed['original_language'].value_counts().nlargest(10)
                sns.barplot(x=top_langs.index, y=top_langs.values, ax=ax_lang)
                ax_lang.set_title("Top 10 Idiomas Originales")
                ax_lang.set_ylabel("Cantidad de Películas")
                st.pyplot(fig_lang)
            else:
                st.write("No hay datos suficientes para `original_language`.")

    else:
        st.warning("El DataFrame procesado no está disponible para análisis.")


elif selected_stage == "Preparación de Datos para Modelado (Entrega 2)":
    st.header("5. Preparación de Datos para Modelado (Segunda Entrega)")
    if df_processed is not None:
        st.markdown("Continuamos con el `Dataset Procesado`.")

        # Re-seleccionar target y features si es necesario o tomar de la etapa anterior
        st.markdown("### Definición de X (Características) e y (Objetivo)")

        all_possible_features = df_processed.columns.tolist()

        # Variable Objetivo
        default_target_idx = 0
        if 'revenue' in df_processed.columns:
            default_target_idx = df_processed.columns.get_loc('revenue')

        target_variable_prep = st.selectbox("Variable Objetivo (y):",
                                            df_processed.columns,
                                            index=default_target_idx,
                                            key="target_prep")

        # Selección de Características
        potential_features = [col for col in df_processed.columns if col != target_variable_prep]

        # Default features (ejemplo, podrían venir de la etapa de análisis)
        default_feature_selection = [
            'budget', 'popularity', 'runtime', 'vote_count', 'release_year',
            'num_genres', 'num_production_companies', 'original_language'  # Incluyendo una categórica
        ]
        default_feature_selection = [f for f in default_feature_selection if f in potential_features]

        selected_features_prep = st.multiselect("Características (X):",
                                                potential_features,
                                                default=default_feature_selection,
                                                key="features_prep")

        if not target_variable_prep or not selected_features_prep:
            st.warning("Por favor, selecciona la variable objetivo y al menos una característica.")
            st.stop()

        X = df_processed[selected_features_prep]
        y = df_processed[target_variable_prep]

        # Manejo de NaNs residuales en X o y (importante antes de modelar)
        X.dropna(inplace=True)  # O imputar
        y = y[X.index]  # Asegurar alineación después de dropear NaNs en X

        if X.empty or y.empty:
            st.error(
                "Después de manejar NaNs, no quedan datos suficientes. Revisa la limpieza de las columnas seleccionadas.")
            st.stop()

        st.markdown("### Preprocesamiento (Codificación y Escalado)")

        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Crear transformadores
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore',
                                                sparse_output=False)  # sparse_output=False para ver array

        # Crear preprocesador con ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='passthrough'  # 'passthrough' o 'drop' para columnas no especificadas
        )

        st.markdown(f"**Características Numéricas a Escalar:** {numeric_features}")
        st.markdown(f"**Características Categóricas a Codificar (One-Hot):** {categorical_features}")

        st.markdown("### División en Conjuntos de Entrenamiento y Prueba")
        test_size = st.slider("Tamaño del conjunto de prueba (test_size):", 0.1, 0.5, 0.2, 0.05, key="test_size_slider")
        random_state = st.number_input("Semilla aleatoria (random_state):", value=42, key="random_state_input")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        st.write(f"Forma de X_train: {X_train.shape}, Forma de X_test: {X_test.shape}")
        st.write(f"Forma de y_train: {y_train.shape}, Forma de y_test: {y_test.shape}")

        # Aplicar preprocesamiento (solo fit_transform en train, transform en test)
        # Esto se hará dentro del pipeline del modelo para evitar data leakage

        # Guardar en session state para la etapa de modelado
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.preprocessor = preprocessor
        st.session_state.selected_features = selected_features_prep  # Guardar las features usadas

        st.success("Datos preparados y divididos. Listos para el modelado.")

    else:
        st.warning("El DataFrame procesado no está disponible.")


elif selected_stage == "Modelado y Evaluación (Entrega 2)":
    st.header("6. Modelado y Evaluación de Modelos (Segunda Entrega)")

    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning(
            "Los datos de entrenamiento no están disponibles. Por favor, completa la etapa 'Preparación de Datos para Modelado' primero.")
        st.stop()

    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    preprocessor = st.session_state.preprocessor

    st.markdown(f"Usando las características: `{st.session_state.selected_features}`")
    st.markdown(f"Variable objetivo: `{y_train.name}` (asumiendo que y_train es una Serie con nombre)")

    # Selección de Modelos
    st.markdown("### Selección y Entrenamiento de Modelos de Regresión")
    model_options = {
        "Regresión Lineal": LinearRegression(),
        "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100)
        # n_estimators es un hiperparámetro común
    }

    selected_model_names = st.multiselect("Selecciona los modelos a entrenar:",
                                          list(model_options.keys()),
                                          default=list(model_options.keys())[:2],  # Por defecto los dos primeros
                                          key="model_select_multi")

    if not selected_model_names:
        st.warning("Por favor, selecciona al menos un modelo.")
        st.stop()

    results = {}

    if st.button("Entrenar y Evaluar Modelos Seleccionados", key="train_eval_button"):
        with st.spinner("Entrenando y evaluando modelos..."):
            for model_name in selected_model_names:
                model = model_options[model_name]

                # Crear pipeline con preprocesador y modelo
                pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                           ('regressor', model)])

                # Entrenar
                pipeline.fit(X_train, y_train)

                # Predecir
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)

                # Evaluar
                r2_train = r2_score(y_train, y_pred_train)
                mse_train = mean_squared_error(y_train, y_pred_train)
                mae_train = mean_absolute_error(y_train, y_pred_train)

                r2_test = r2_score(y_test, y_pred_test)
                mse_test = mean_squared_error(y_test, y_pred_test)
                rmse_test = np.sqrt(mse_test)
                mae_test = mean_absolute_error(y_test, y_pred_test)

                results[model_name] = {
                    "R² (Train)": r2_train, "MSE (Train)": mse_train, "MAE (Train)": mae_train,
                    "R² (Test)": r2_test, "MSE (Test)": mse_test, "RMSE (Test)": rmse_test, "MAE (Test)": mae_test,
                    "y_pred_test": y_pred_test  # Guardar predicciones para gráficos
                }

        st.session_state.model_results = results  # Guardar resultados en session state

    # Mostrar Resultados si existen
    if 'model_results' in st.session_state and st.session_state.model_results:
        results_df = pd.DataFrame(st.session_state.model_results).T  # Transponer para mejor visualización
        st.subheader("Métricas de Evaluación de los Modelos")
        st.dataframe(results_df[['R² (Test)', 'RMSE (Test)', 'MAE (Test)', 'R² (Train)']].style.format("{:.4f}"))

        st.subheader("Visualización de Predicciones vs. Valores Reales (Test Set)")
        for model_name, result_data in st.session_state.model_results.items():
            if model_name in selected_model_names:  # Solo mostrar los que se acaban de (re)calcular
                st.markdown(f"#### {model_name}")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, result_data["y_pred_test"], alpha=0.5, label="Predicciones")
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', lw=2,
                        label="Línea Ideal (y=x)")
                ax.set_xlabel(f"Valores Reales ({y_test.name})")
                ax.set_ylabel(f"Valores Predichos ({y_test.name})")
                ax.set_title(f"Predicciones vs. Reales para {model_name}")
                ax.legend()
                st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Fin de las opciones.")

