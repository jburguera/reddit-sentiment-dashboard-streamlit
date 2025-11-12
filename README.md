# Tesla Sentiment Dashboard

Dashboard interactivo para analizar el sentiment sobre Tesla en Reddit usando procesamiento de lenguaje natural y visualización de datos.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-green?style=for-the-badge)

## Características principales

- Recopilación de datos de Reddit desde cualquier subreddit
- Análisis de sentiment con VADER (NLTK)
- Gráficos interactivos con Plotly y Matplotlib
- Modelado de tópicos con LDA para descubrir temas de conversación
- Nubes de palabras segmentadas por sentiment
- Detección automática de tendencias
- Exportación de datos en CSV y JSON

## Qué incluye

### Distribución de sentiment
- Gráfico circular con distribución positiva/neutral/negativa
- Histograma de puntuaciones
- Violin plots para ver la distribución en detalle

### Análisis temporal
- Tendencias de sentiment en el tiempo
- Patrones por hora del día
- Volumen de comentarios

### Posts destacados
- Posts más positivos y negativos con enlaces directos
- Posts más discutidos
- Métricas de engagement (upvotes, comentarios, ratios)

### Análisis de palabras
- Nubes de palabras por cada tipo de sentiment
- Top 15 palabras más comunes
- Modelado de tópicos interactivo
- Puntuación de coherencia para validar la calidad

### Exportación de datos
- Descargar comentarios en CSV
- Descargar posts en CSV
- Reporte JSON completo con todas las métricas

## Instalación

### Requisitos

- Python 3.8 o superior
- Credenciales de API de Reddit (gratis)

### Pasos

1. Clonar el repositorio:
```bash
git clone https://github.com/yourusername/reddit-sentiment-dashboard-streamlit.git
cd reddit-sentiment-dashboard-streamlit
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Obtener credenciales de Reddit:
   - Ir a https://www.reddit.com/prefs/apps
   - Click en "Create App"
   - Seleccionar "script" como tipo de app
   - Anotar el `client_id` y `client_secret`

4. Crear el archivo `reddit_credentials.txt`:
```
CLIENT_ID=tu_client_id_aqui
CLIENT_SECRET=tu_client_secret_aqui
USER_AGENT=nombre_de_tu_app/1.0
```

5. Ejecutar:
```bash
streamlit run streamlit_app.py
```

El dashboard se abrirá en `http://localhost:8501`

## Uso

1. Configurar parámetros en el sidebar:
   - Elegir el subreddit (sin el r/)
   - Número de posts a analizar
   - Filtro de mínimo de comentarios
   - Período de tiempo

2. Click en "🔍 Analyze Sentiment"

3. Explorar las visualizaciones y exportar datos si es necesario

## Modelado de tópicos

El dashboard usa LDA (Latent Dirichlet Allocation) para descubrir temas automáticamente:
- Ajusta el número de tópicos (2-10)
- Puntuaciones de coherencia entre 0.4-0.7 indican buenos resultados
- Click en los tópicos en la visualización interactiva para explorar

## Detección de tendencias

El análisis de tendencias usa promedios móviles:
- "Up": >5% de aumento en sentiment
- "Down": >5% de caída en sentiment
- "Stable": cambios dentro de ±5%

## Dependencias

- streamlit - Framework web
- matplotlib - Gráficos estáticos
- seaborn - Visualización estadística
- pandas - Manipulación de datos
- praw - API de Reddit
- nltk - Análisis de sentiment
- wordcloud - Nubes de palabras
- gensim - Modelado de tópicos
- pyLDAvis - Visualización LDA
- plotly - Gráficos interactivos
- statsmodels - Líneas de tendencia

## Estructura del proyecto

```
reddit-sentiment-dashboard-streamlit/
├── streamlit_app.py           # Archivo principal
├── requirements.txt           # Dependencias
├── reddit_credentials.txt     # Credenciales (crear este archivo)
├── README.md
└── nltk_data/                 # Datos NLTK (se crea automáticamente)
```

## Pipeline de procesamiento

1. Preprocesamiento de texto:
   - Convertir a minúsculas
   - Eliminar URLs y caracteres especiales
   - Tokenización
   - Eliminar stopwords (incluyendo palabras específicas de Reddit)
   - Lemmatización

2. Análisis de sentiment:
   - VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Puntuación compuesta de -1 (muy negativo) a +1 (muy positivo)
   - Clasificación: Positivo (≥0.05), Neutral (-0.05 a 0.05), Negativo (≤-0.05)

3. Modelado de tópicos:
   - LDA con ajuste automático de parámetros
   - Cálculo de coherencia (métrica C_v)
   - Visualización interactiva
   - Filtrado de valores extremos

## Solución de problemas

### Recursos NLTK no encontrados
La app descarga automáticamente los datos necesarios. Si hay problemas:
```bash
python -c "import nltk; nltk.download('all')"
```

### Límite de API de Reddit
- Reducir el número de posts a analizar
- Aumentar el filtro de mínimo de comentarios
- Esperar unos minutos antes de hacer nuevas peticiones

### Errores de modelado de tópicos
- Asegúrate de tener al menos 10 documentos válidos
- Intenta analizar más posts
- Ajusta el número de tópicos

### Problemas de memoria
- Reduce el límite de posts
- Limpia la caché desde el menú de Streamlit
- Reinicia la app

## Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles

## Créditos

- NLTK para análisis de lenguaje natural
- PRAW para la API de Reddit
- Streamlit para el framework web
- Plotly para visualizaciones interactivas
- Gensim para modelado de tópicos

---

Hecho con Python y Streamlit
