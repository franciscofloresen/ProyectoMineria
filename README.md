# 🎬 **Proyecto de Minería de Datos - Análisis de Películas**

## 📌 **Descripción del Proyecto**
Este proyecto tiene como objetivo la **preparación y modelado de datos de películas**, aplicando técnicas de minería de datos para mejorar la calidad del dataset y facilitar su análisis.

## 🛠 **Tecnologías Utilizadas**
- 🐍 **Python**
- 📊 `pandas` - Manejo y transformación de datos
- 🔢 `numpy` - Operaciones matemáticas
- 📈 `matplotlib` & `seaborn` - Visualización de datos
- 🌐 `streamlit` - Interfaz interactiva para visualizar cambios en los datos
- 📝 `ast` - Conversión de datos almacenados como cadenas

## 📂 **Estructura del Código**

### 🔹 **1. Carga del Dataset**
📥 Se lee el archivo **`TMDB_movie_dataset_v11.csv`** y se almacena en un DataFrame.

### 🔹 **2. Limpieza de Datos**
✅ Eliminación de columnas irrelevantes: `homepage`, `poster_path`, `backdrop_path`, `tagline`, `imdb_id`  
✅ Imputación de valores nulos en `budget`, `revenue` y `runtime` con la **mediana**  
✅ Eliminación de películas con menos de **10 votos** para evitar sesgos  

### 🔹 **3. Transformación de Datos**
⚡ **Normalización**: `vote_average` escalado entre 0 y 1  
📅 **Conversión de fechas**: Se extrae `release_year` de `release_date`  
📊 **Eliminación de valores extremos** en `budget` y `revenue`  

### 🔹 **4. Visualización con Streamlit**
📊 **Comparación de `vote_average` antes y después de la normalización**  
📉 **Distribución de valores antes y después de la limpieza**  

### 🔹 **5. Guardado del Dataset Procesado**
💾 Se genera un nuevo archivo: **`peliculas_procesadas.csv`**

---

## 🚀 **Ejecución del Código**

### 🔹 **1️⃣ Instalar las dependencias necesarias**
```bash
pip install pandas numpy matplotlib seaborn streamlit
```

### 🔹 **2️⃣ Ejecutar el script en Streamlit**
```bash
streamlit run FinalApp.py
```

### 🔹 **3️⃣ Explorar la interfaz interactiva**
🔍 Visualizar gráficos antes y después de la limpieza  
📊 Analizar transformaciones aplicadas a las variables  
📂 Ver el dataset final optimizado  

---

## 📑 **Archivos Generados**
📂 **`peliculas_procesadas.csv`** - Dataset limpio y optimizado  
📄 **`Presentacion_Proyecto_Final.pptx`** - Resumen del proceso y hallazgos principales  
📜 **`ETL.py`** - Código en Python con la ejecución completa de la limpieza y transformación de datos  

---

## 📌 **Conclusión**
✔️ Se mejoró la calidad del dataset mediante limpieza y transformación  
✔️ Se eliminaron valores irrelevantes y se corrigieron inconsistencias  
✔️ Se aplicó **normalización y extracción de datos clave** para facilitar análisis futuros  

💡 **Con estos datos optimizados, se pueden realizar análisis más precisos y aplicar modelos predictivos en futuras fases del proyecto.** 🎯

