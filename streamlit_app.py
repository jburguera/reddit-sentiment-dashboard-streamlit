import nltk
try:
    nltk.download('vader_lexicon')
    print("Recurso NLTK 'vader_lexicon' descargado exitosamente.") # Añadir mensaje para verificar en logs (opcional)
except Exception as e:
    print(f"Error al descargar recurso NLTK 'vader_lexicon': {e}") # Manejar error en la descarga (opcional)
    # Considerar si quieres que la app se detenga si falla la descarga (opcional)
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Inicializar analizador VADER (para Streamlit)
analyzer_reddit_vader_streamlit = SentimentIntensityAnalyzer()

# Configuración de la página Streamlit
st.set_page_config(page_title="Reddit Sentiment Dashboard - TeslaMotors", page_icon="📈", layout="wide")

# Título del Dashboard
st.title("Dashboard de sentimiento en Reddit: Opinión pública sobre Tesla en r/TeslaMotors")

# Descripción del Dashboard
st.markdown("""
Dashboard que visualiza el sentimiento público hacia Tesla en el subreddit **r/TeslaMotors**,
utilizando análisis de sentimiento con **VADER** de NLTK.
""")

st.markdown("---")

import os

# Cargar credenciales de Reddit desde archivo externo 'reddit_credentials.txt'
try:
    credentials_reddit = {}
    with open("reddit_credentials.txt", "r") as f:
        for line in f:
            key, value = line.strip().split("=", 1)
            credentials_reddit[key.strip()] = value.strip()

    CLIENT_ID = credentials_reddit["CLIENT_ID"]
    CLIENT_SECRET = credentials_reddit["CLIENT_SECRET"]
    USER_AGENT = credentials_reddit["USER_AGENT"]

    print("Credenciales de Reddit cargadas.")

except FileNotFoundError:
    print("Error: Archivo 'reddit_credentials.txt' no encontrado.")
    os.abort() # Terminar la ejecución si no se encuentran las credenciales
except KeyError:
    print("Error: Credenciales incompletas en 'reddit_credentials.txt'.")
    os.abort() # Terminar la ejecución si faltan credenciales

# Conexión a la API de Reddit

import praw

# Inicializar instancia de PRAW Reddit API client
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    print("Conexión a la API de Reddit establecida.")

except Exception as e:
    print(f"Error al conectar con la API de Reddit: {e}")
    os.abort() # Terminar la ejecución si falla la conexión

# Recopilación de Publicaciones y Comentarios

# Configuración de parámetros de recopilación
SUBREDDIT_NAME = "TeslaMotors"
MIN_COMMENTS = 150
NUM_POSTS_FETCH = 200
NUM_POSTS_DISPLAY = 10 # Límite para visualización en salida de Colab (no afecta al dashboard completo)

# Listas para almacenar datos
post_data_list = []
comment_data_list = []

print(f"Recopilando publicaciones y comentarios de r/{SUBREDDIT_NAME}...")

try:
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
    hot_posts_initial = subreddit.hot(limit=NUM_POSTS_FETCH)
    filtered_posts = [post for post in hot_posts_initial if post.num_comments >= MIN_COMMENTS]

    for post in filtered_posts[:NUM_POSTS_DISPLAY]: # Límite para eficiencia en Colab
        post_data = { # Diccionario para datos de publicación
            "post_id": post.id, "post_title": post.title, "post_text": post.selftext,
            "post_url": f"https://www.reddit.com{post.permalink}", "post_score": post.score,
            "post_upvote_ratio": post.upvote_ratio, "post_num_comments": post.num_comments,
            "post_created_utc": post.created_utc, "post_author": str(post.author)
        }
        post_data_list.append(post_data)

        post.comments.replace_more(limit=0) # Cargar comentarios de primer nivel
        for comment in post.comments:
            comment_data = { # Diccionario para datos de comentario
                "comment_id": comment.id, "comment_text": comment.body, "comment_author": str(comment.author),
                "comment_score": comment.score, "comment_created_utc": comment.created_utc,
                "post_id": post.id, "post_title": post.title
            }
            comment_data_list.append(comment_data)

    print(f"Recopilación completada: {len(post_data_list)} publicaciones, {len(comment_data_list)} comentarios.")

except Exception as e:
    print(f"Error durante la recopilación de datos: {e}")


# --- Código de procesamiento de datos (incluido para que Streamlit funcione de forma autónoma) ---
# (Incluye aquí los Pasos 5, 6 y 7.1 como en el código completo anterior)
# DataFrame para publicaciones y comentarios (reutilizando post_data_list y comment_data_list de la celda anterior - ¡importante!)
df_posts_reddit_streamlit = pd.DataFrame(post_data_list) # Reutilizar post_data_list
df_comments_reddit_streamlit = pd.DataFrame(comment_data_list) # Reutilizar comment_data_list


# Análisis de Sentimiento VADER (reutilizando df_comments_reddit_streamlit)
df_comments_reddit_streamlit['vader_neg'] = None # Inicializar columnas (necesario en entorno Streamlit)
df_comments_reddit_streamlit['vader_neu'] = None
df_comments_reddit_streamlit['vader_pos'] = None
df_comments_reddit_streamlit['vader_compound'] = None


for index, row in df_comments_reddit_streamlit.iterrows(): # Reutilizar df_comments_reddit_streamlit
    comment_text = row['comment_text']
    if isinstance(comment_text, str):
        vs = analyzer_reddit_vader_streamlit.polarity_scores(comment_text) # Reutilizar analyzer_reddit_vader_streamlit
        df_comments_reddit_streamlit.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = vs.values()
    else:
        df_comments_reddit_streamlit.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = [0.0, 1.0, 0.0, 0.0]


# Preparación de datos para visualización (reutilizando df_comments_reddit_streamlit)
def categorize_sentiment_streamlit(compound_score): # Función para Streamlit
    if compound_score >= 0.05: return "Positivo"
    elif compound_score <= -0.05: return "Negativo"
    else: return "Neutral"

df_comments_reddit_streamlit['sentiment_category'] = df_comments_reddit_streamlit['vader_compound'].apply(categorize_sentiment_streamlit) # Reutilizar df_comments_reddit_streamlit
sentiment_counts_streamlit = df_comments_reddit_streamlit['sentiment_category'].value_counts() # Reutilizar df_comments_reddit_streamlit
average_compound_score_streamlit = df_comments_reddit_streamlit['vader_compound'].mean() # Reutilizar df_comments_reddit_streamlit


# --- Visualizaciones en Streamlit ---

# Gráfico de Barras de Categorías de Sentimiento
st.subheader("Distribución de Categorías de Sentimiento")
fig_bar_chart, ax_bar_chart = plt.subplots(figsize=(8, 6))
sns.barplot(x=sentiment_counts_streamlit.index, y=sentiment_counts_streamlit.values, palette="viridis", ax=ax_bar_chart) # Reutilizar sentiment_counts_streamlit
ax_bar_chart.set_title('Distribución de Categorías de Sentimiento')
ax_bar_chart.set_xlabel('Categoría de Sentimiento')
ax_bar_chart.set_ylabel('Número de Comentarios')
ax_bar_chart.tick_params(axis='x', rotation=45)
st.pyplot(fig_bar_chart)


# Histograma de Puntuaciones Compuestas de Sentimiento
st.subheader("Distribución de Puntuaciones Compuestas de Sentimiento (VADER)")
fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
sns.histplot(df_comments_reddit_streamlit['vader_compound'], bins=30, kde=True, color='skyblue', ax=ax_hist) # Reutilizar df_comments_reddit_streamlit
ax_hist.set_title('Distribución de Puntuaciones Compuestas de Sentimiento (VADER)')
ax_hist.set_xlabel('Puntuación Compuesta de Sentimiento (VADER)')
ax_hist.set_ylabel('Frecuencia')
ax_hist.set_xlim(-1, 1)
st.pyplot(fig_hist)
