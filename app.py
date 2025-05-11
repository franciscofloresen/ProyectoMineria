import streamlit as st
import pandas as pd
import numpy as np
import ast  # Para convertir strings de listas/diccionarios a objetos Python
import json  # Para manejar estructuras JSON complejas
from io import StringIO

# --- Configuración de la Página ---
st.set_page_config(layout="wide", page_title="Proyecto Minería de Datos: Películas TMDB - Entrega 1")

# --- RUTA DEL DATASET ---
DATASET_PATH = "data/TMDB_movie_dataset_v11.csv"  # Asegúrate que este archivo exista en la ruta o ajústala


# --- Funciones Auxiliares ---

def load_data(file_path):
    """Carga los datos desde una ruta de archivo CSV."""
    try:
        df = pd.read_csv(file_path)
        st.success(f"Dataset cargado exitosamente desde: {file_path}")
        return df
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo en la ruta especificada: {file_path}")
        st.error(
            "Asegúrate de que el archivo TMDB_movie_dataset_v11.csv esté en la misma carpeta que el script, o actualiza la variable DATASET_PATH en el código.")
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
    st.dataframe(df.describe(include='all'))


def display_missing_values(df, title="Valores Faltantes"):
    """Muestra un resumen de valores faltantes."""
    st.subheader(title)
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]  # Solo mostrar columnas con nulos
    if not missing_data.empty:
        st.write("Conteo de valores faltantes por columna:")
        st.dataframe(missing_data.sort_values(ascending=False))
        st.bar_chart(missing_data.sort_values(ascending=False))
    else:
        st.success("¡No hay valores faltantes en el dataset!")


def safe_literal_eval(val):
    """Intenta convertir un string a un objeto Python (lista/diccionario).
       Maneja errores comunes y valores NaN."""
    if pd.isna(val):
        return []  # Retorna lista vacía para NaNs
    try:
        # Intenta reemplazar comillas simples problemáticas antes de evaluar
        # Esto es un intento básico, puede necesitar ajustes más robustos
        if isinstance(val, str):
            val = val.replace("\\'", "'")  # Escapar comillas simples dentro de strings
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        # st.warning(f"No se pudo parsear con ast.literal_eval: {val}. Se devuelve lista vacía.")
        return []
    except Exception:
        # st.warning(f"Excepción desconocida al parsear: {val}. Se devuelve lista vacía.")
        return []


def extract_names_from_json_like_list(series, key_name='name'):
    """Extrae una lista de nombres de una columna que contiene strings de listas de diccionarios."""
    all_items = []
    for _, row_str in series.dropna().items():  # Iterar sobre las filas no nulas
        try:
            list_of_dicts = safe_literal_eval(row_str)
            if isinstance(list_of_dicts, list):
                names = [d[key_name] for d in list_of_dicts if isinstance(d, dict) and key_name in d]
                all_items.append(names)
            else:
                all_items.append([])
        except Exception:
            all_items.append([])

    extracted_series = pd.Series(index=series.index, dtype='object')
    valid_indices = series.dropna().index

    if len(all_items) == len(valid_indices):
        extracted_series.loc[valid_indices] = all_items
    else:
        # Fallback si hay discrepancia, aunque no debería ocurrir con la lógica actual
        for idx in valid_indices:
            extracted_series.loc[idx] = []

    extracted_series = extracted_series.fillna("[]").apply(lambda x: x if isinstance(x, list) else [])
    return extracted_series


# --- Título de la Aplicación ---
st.title("🎬 Proyecto Minería de Datos: Análisis de Películas TMDB")
st.header("Primera Entrega: Entendimiento y Preparación de Datos")

# --- Carga de Datos ---
st.sidebar.title("Navegación")
st.sidebar.header("1. Carga de Datos")
st.sidebar.info(f"Intentando cargar dataset desde: {DATASET_PATH}")

df_original = load_data(DATASET_PATH)

if df_original is None:
    st.error("El DataFrame no pudo ser cargado. Por favor, revisa la ruta del archivo y los mensajes de error.")
    st.stop()

df = df_original.copy()

# --- Etapa 1: Entendimiento del Negocio (Texto Descriptivo) ---
st.sidebar.markdown("---")
st.sidebar.header("Etapas CRISP-DM")
selected_stage = st.sidebar.radio("Selecciona la etapa a visualizar:",
                                  ["Entendimiento del Negocio",
                                   "Entendimiento de los Datos",
                                   "Preparación de los Datos"])
st.sidebar.markdown("---")

if selected_stage == "Entendimiento del Negocio":
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

# --- Etapa 2: Entendimiento de los Datos ---
elif selected_stage == "Entendimiento de los Datos":
    st.header("2. Entendimiento de los Datos")
    st.markdown(f"Dataset cargado desde: `{DATASET_PATH}`")
    st.markdown(f"Número de filas: `{df.shape[0]}`")
    st.markdown(f"Número de columnas: `{df.shape[1]}`")

    display_dataframe_info(df, "Información Inicial del Dataset")
    display_missing_values(df, "Valores Faltantes Iniciales")

    st.subheader("Exploración de Columnas Específicas")
    column_to_explore = st.selectbox("Selecciona una columna para ver sus valores únicos (si son pocos) o ejemplos:",
                                     df.columns)
    if column_to_explore and column_to_explore in df.columns:  # Verificar que la columna exista
        if df[column_to_explore].nunique() < 20 and df[column_to_explore].nunique() > 0:
            st.write(f"Valores únicos en '{column_to_explore}':")
            st.write(df[column_to_explore].unique())
        else:
            st.write(f"Algunos ejemplos de valores en '{column_to_explore}':")
            st.dataframe(df[column_to_explore].sample(min(5, len(df))))

        json_like_cols = ['genres', 'production_companies', 'production_countries', 'spoken_languages', 'keywords']
        if column_to_explore in json_like_cols:
            st.write(f"Ejemplo del formato de '{column_to_explore}' (primera fila no nula):")
            first_valid_value = df[column_to_explore].dropna().iloc[0] if not df[
                column_to_explore].dropna().empty else "No hay valores no nulos"
            st.code(str(first_valid_value))  # Convertir a string para st.code


# --- Etapa 3: Preparación de los Datos ---
elif selected_stage == "Preparación de los Datos":
    st.header("3. Preparación de los Datos")

    df_processed = df.copy()

    st.subheader("3.1. Limpieza y Manejo de Valores Nulos")

    # --- Presupuesto (budget) y Ingresos (revenue) ---
    st.markdown("**Presupuesto (`budget`) e Ingresos (`revenue`):**")
    df_processed['budget'] = pd.to_numeric(df_processed['budget'], errors='coerce')
    df_processed['revenue'] = pd.to_numeric(df_processed['revenue'], errors='coerce')
    # Contar ceros y nulos después de la conversión
    budget_zeros_or_nulls = (df_processed['budget'] == 0).sum() + df_processed['budget'].isnull().sum()
    revenue_zeros_or_nulls = (df_processed['revenue'] == 0).sum() + df_processed['revenue'].isnull().sum()
    st.markdown(f"Películas con `budget` = 0 o nulo (tras conversión): {budget_zeros_or_nulls}")
    st.markdown(f"Películas con `revenue` = 0 o nulo (tras conversión): {revenue_zeros_or_nulls}")
    st.markdown("""
    Los valores de 0 en `budget` y `revenue` (o los que se convirtieron a nulos si no eran numéricos) pueden ser datos faltantes o ceros reales. 
    Para este ejercicio inicial, los dejaremos como están después de la conversión a numérico, pero en un análisis más profundo,
    se deberían investigar e imputar si es necesario, especialmente si son predictores o la variable objetivo.
    """)

    # --- Duración (runtime) ---
    st.markdown("**Duración (`runtime`):**")
    df_processed['runtime'] = pd.to_numeric(df_processed['runtime'], errors='coerce')
    original_runtime_nans = df_processed['runtime'].isnull().sum()
    if original_runtime_nans > 0:
        median_runtime = df_processed['runtime'].median()
        if pd.notna(median_runtime):
            df_processed['runtime'].fillna(median_runtime, inplace=True)
            st.write(
                f"Se imputaron `{original_runtime_nans}` valores nulos/no numéricos en `runtime` con la mediana: `{median_runtime:.2f}` minutos.")
        else:
            st.warning("No se pudo calcular la mediana para `runtime`. `runtime` no fue imputado.")
    else:
        st.write("No hay valores nulos o no numéricos problemáticos en `runtime`.")

    # --- Fecha de Estreno (release_date) ---
    st.markdown("**Fecha de Estreno (`release_date`):**")
    # Guardar el número de filas antes de cualquier eliminación por release_date
    rows_before_rd_cleaning = len(df_processed)

    # Eliminar filas si 'release_date' es NaN o string vacío antes de convertir
    df_processed.dropna(subset=['release_date'], inplace=True)
    # Asegurarse que release_date es string antes de comparar con ''
    df_processed = df_processed[df_processed['release_date'].astype(str) != '']

    # Convertir a datetime
    df_processed['release_date'] = pd.to_datetime(df_processed['release_date'], errors='coerce')

    # Eliminar filas donde la conversión a datetime falló (resultando en NaT)
    df_processed.dropna(subset=['release_date'], inplace=True)

    rows_after_rd_cleaning = len(df_processed)
    rows_removed_rd = rows_before_rd_cleaning - rows_after_rd_cleaning
    if rows_removed_rd > 0:
        st.write(
            f"Se eliminaron `{rows_removed_rd}` filas donde `release_date` era nulo, vacío o no pudo ser convertida a fecha válida.")
    else:
        st.write("No se eliminaron filas debido a problemas con `release_date`.")

    # --- Columnas de texto como overview, tagline ---
    st.markdown("**Columnas de Texto (`overview`, `tagline`):**")
    text_cols_to_fill = ['overview', 'tagline']
    for col in text_cols_to_fill:
        if col in df_processed.columns:
            original_text_nans = df_processed[col].isnull().sum()
            if original_text_nans > 0:
                df_processed[col].fillna("", inplace=True)
                st.write(f"Se imputaron `{original_text_nans}` valores nulos en `{col}` con un string vacío.")

    st.subheader("3.2. Transformación de Datos")

    # --- Extracción de Año, Mes, Día de la Semana de release_date ---
    st.markdown("**Extracción de Componentes de Fecha:**")
    if 'release_date' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['release_date']):
        df_processed['release_year'] = df_processed['release_date'].dt.year
        df_processed['release_month'] = df_processed['release_date'].dt.month
        df_processed['release_dayofweek'] = df_processed['release_date'].dt.dayofweek
        st.write("Se crearon las columnas: `release_year`, `release_month`, `release_dayofweek`.")
        if not df_processed.empty:
            st.dataframe(
                df_processed[['title', 'release_date', 'release_year', 'release_month', 'release_dayofweek']].head(
                    min(5, len(df_processed))))
    else:
        st.warning(
            "La columna `release_date` no es de tipo fecha válida o no existe después de la limpieza. No se pudieron extraer componentes.")

    # --- Procesamiento de columnas JSON-like (genres, production_companies, etc.) ---
    st.markdown("**Procesamiento de Columnas con formato JSON/Lista:**")
    json_like_cols_data = {
        'genres': 'name',
        'production_companies': 'name',
        'production_countries': 'name',  # o 'iso_3166_1'
        'spoken_languages': 'english_name',  # o 'iso_639_1'
        'keywords': 'name'
    }

    # Guardar las columnas que se crearán para luego eliminarlas
    temp_cols_to_drop = []

    for col_name, key_for_name in json_like_cols_data.items():
        if col_name in df_processed.columns:
            st.write(f"Procesando `{col_name}`...")
            df_processed[col_name].fillna('[]', inplace=True)

            list_col = f'{col_name}_list'
            first_col = f'first_{col_name}'
            num_col = f'num_{col_name}'

            temp_cols_to_drop.extend([list_col, first_col, num_col])

            extracted_names = extract_names_from_json_like_list(df_processed[col_name], key_name=key_for_name)
            df_processed[list_col] = extracted_names

            df_processed[first_col] = df_processed[list_col].apply(
                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
            df_processed[num_col] = df_processed[list_col].apply(lambda x: len(x) if isinstance(x, list) else 0)

            st.write(f" - Temporalmente se creó `{list_col}` con la lista de nombres.")
            st.write(f" - Temporalmente se creó `{first_col}` con el primer nombre de la lista.")
            st.write(f" - Temporalmente se creó `{num_col}` con la cantidad de elementos.")

            # Mostrar una muestra de las columnas temporales creadas
            # if not df_processed.empty and all(c in df_processed.columns for c in ['title', col_name, list_col, first_col, num_col]):
            #     st.dataframe(df_processed[['title', col_name, list_col, first_col, num_col]].head(min(3, len(df_processed))))
            # else:
            #     st.warning(f"No se pudieron mostrar las columnas derivadas temporalmente para {col_name}, o el dataframe está vacío/columnas faltantes.")
        else:
            st.warning(f"La columna `{col_name}` no se encuentra en el DataFrame.")

    st.subheader("3.3. Creación de Nuevas Características (Feature Engineering Básico)")
    st.markdown("**Cálculo de `profit` y `return_on_investment (ROI)`:**")
    if 'revenue' in df_processed.columns and 'budget' in df_processed.columns:
        df_processed['profit'] = df_processed['revenue'] - df_processed['budget']

        df_processed['roi'] = np.where(
            (df_processed['budget'] > 0) & pd.notna(df_processed['budget']) & pd.notna(df_processed['profit']),
            (df_processed['profit'] / df_processed['budget']) * 100,
            0
        )
        st.write("Se crearon las columnas `profit` y `roi`.")
        if not df_processed.empty and all(
                c in df_processed.columns for c in ['title', 'budget', 'revenue', 'profit', 'roi']):
            st.dataframe(df_processed[['title', 'budget', 'revenue', 'profit', 'roi']].head(min(5, len(df_processed))))
        else:
            st.warning("No se pudieron mostrar `profit` y `roi`, o el dataframe está vacío/columnas faltantes.")
    else:
        st.warning(
            "Las columnas `revenue` o `budget` no están disponibles o no son numéricas para calcular `profit` y `roi`.")

    st.subheader("3.4. Selección/Eliminación de Columnas")

    columns_to_remove_user_specified = [
        'genres_list', 'first_genres', 'num_genres',
        'production_companies_list', 'first_production_companies', 'num_production_companies',
        'production_countries_list', 'first_production_countries', 'num_production_countries',
        'spoken_languages_list', 'first_spoken_languages', 'num_spoken_languages',
        'keywords_list', 'first_keywords', 'num_keywords'
    ]

    # Combinar con otras columnas que podrían ser candidatas a eliminar (ejemplo)
    other_cols_to_drop_example = ['id', 'imdb_id', 'original_title', 'overview', 'tagline',
                                  'poster_path', 'backdrop_path', 'homepage',
                                  'genres', 'production_companies', 'production_countries',
                                  'spoken_languages', 'keywords']  # Las JSON originales, si ya no se necesitan

    # Lista final de columnas a eliminar (solo las especificadas por el usuario por ahora)
    # Si quieres eliminar las JSON originales también, descomenta la línea de 'other_cols_to_drop_example'
    # y combínalas: all_cols_to_drop = list(set(columns_to_remove_user_specified + other_cols_to_drop_example))
    all_cols_to_drop = columns_to_remove_user_specified

    # Filtrar para solo intentar eliminar columnas que existen
    existing_cols_to_drop = [col for col in all_cols_to_drop if col in df_processed.columns]

    if existing_cols_to_drop:
        df_processed.drop(columns=existing_cols_to_drop, inplace=True, errors='ignore')
        st.markdown(
            f"Se eliminaron las siguientes columnas del DataFrame procesado: `{', '.join(existing_cols_to_drop)}`")
    else:
        st.markdown("No se encontraron las columnas especificadas para eliminar en el DataFrame procesado.")

    st.markdown("""
    Además de las columnas derivadas de JSON que se han eliminado, se podrían eliminar otras columnas que no se usarán 
    directamente en el modelado inicial o que ya fueron procesadas (como las columnas JSON originales, identificadores, etc.).
    Por ahora, solo se han eliminado las columnas `*_list`, `first_*`, y `num_*` solicitadas.
    """)

    st.subheader("3.5. Dataset Procesado (Final)")
    st.markdown("""
    A continuación, se muestra el estado del DataFrame después de las transformaciones y la eliminación de columnas.
    Este dataset está más limpio y optimizado para el análisis.
    """)
    display_dataframe_info(df_processed, "Información del Dataset Procesado")
    display_missing_values(df_processed, "Valores Faltantes Después del Procesamiento")

    if not df_processed.empty:
        st.download_button(
            label="Descargar Dataset Procesado (CSV)",
            data=df_processed.to_csv(index=False).encode('utf-8'),
            file_name='TMDB_movies_processed_cleaned.csv',
            mime='text/csv',
        )
    else:
        st.warning("El DataFrame procesado está vacío, no se puede descargar.")

st.sidebar.markdown("---")
st.sidebar.info("Fin de las opciones de la primera entrega.")

