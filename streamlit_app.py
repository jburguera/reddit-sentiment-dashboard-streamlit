import nltk

# Descargar recurso NLTK 'vader_lexicon' para análisis de sentimiento
try:
    nltk.download('vader_lexicon', quiet=True) # Descarga silenciosa para logs más limpios
    print("NLTK 'vader_lexicon' descargado.")
except Exception as e:
    print(f"Error al descargar NLTK 'vader_lexicon': {e}")

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# --- Configuración de la página Streamlit ---
st.set_page_config(page_title="Reddit Sentiment Dashboard - TeslaMotors", page_icon="📈", layout="wide")

# --- Título del Dashboard ---
st.title("Dashboard de Sentimiento en Reddit: Opinión Pública sobre Tesla en r/TeslaMotors")

# --- Descripción del Dashboard ---
st.markdown("""
Dashboard para visualizar el sentimiento público hacia **Tesla** en el subreddit **r/TeslaMotors**,
utilizando análisis de sentimiento con **VADER** de la biblioteca **NLTK**.
""")

st.markdown("---")

# --- Carga de Credenciales de Reddit ---
credentials_reddit = {}
try:
    credentials_reddit = {k.split('=')[0].strip(): k.split('=')[1].strip() for k in open("reddit_credentials.txt", "r") if '=' in k}
    CLIENT_ID = credentials_reddit["CLIENT_ID"]
    CLIENT_SECRET = credentials_reddit["CLIENT_SECRET"]
    USER_AGENT = credentials_reddit["USER_AGENT"]
except FileNotFoundError:
    st.error("Archivo 'reddit_credentials.txt' no encontrado. Debe estar en el directorio raíz del repositorio.")
    st.stop()
except KeyError:
    st.error("Credenciales de Reddit incompletas en 'reddit_credentials.txt'. Revisar el archivo.")
    st.stop()

# --- Conexión a la API de Reddit ---
try:
    reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)
    SUBREDDIT_NAME = "TeslaMotors"
    subreddit = reddit.subreddit(SUBREDDIT_NAME)
except Exception as e:
    st.error(f"Error al conectar con la API de Reddit: {e}")
    st.stop()

# --- Recopilación de Publicaciones y Comentarios ---
MIN_COMMENTS = 150
NUM_POSTS_FETCH = 200
NUM_POSTS_DISPLAY = 10 # Límite para visualización en Colab, no afecta al dashboard completo
post_data_list = []
comment_data_list = []

try:
    hot_posts_initial = subreddit.hot(limit=NUM_POSTS_FETCH)
    filtered_posts = [post for post in hot_posts_initial if post.num_comments >= MIN_COMMENTS]

    for post in filtered_posts[:NUM_POSTS_DISPLAY]: # Límite solo para la muestra en Colab, en Streamlit Cloud se procesan más datos
        post_data = {
            "post_id": post.id, "post_title": post.title, "post_text": post.selftext,
            "post_url": f"https://www.reddit.com{post.permalink}", "post_score": post.score,
            "post_upvote_ratio": post.upvote_ratio, "post_num_comments": post.num_comments,
            "post_created_utc": post.created_utc, "post_author": str(post.author)
        }
        post_data_list.append(post_data)

        post.comments.replace_more(limit=0) # Cargar solo comentarios de primer nivel
        for comment in post.comments:
            comment_data = {
                "comment_id": comment.id, "comment_text": comment.body, "comment_author": str(comment.author),
                "comment_score": comment.score, "comment_created_utc": comment.created_utc,
                "post_id": post.id, "post_title": post.title
            }
            comment_data_list.append(comment_data)


except Exception as e:
    st.error(f"Error durante la recopilación de datos de Reddit: {e}")


# --- Creación de DataFrames de Pandas ---
df_posts_reddit_streamlit = pd.DataFrame(post_data_list)
df_comments_reddit_streamlit = pd.DataFrame(comment_data_list)


# --- Inicializar Analizador de Sentimiento VADER ---
analyzer_reddit_vader_streamlit = SentimentIntensityAnalyzer()

# --- Análisis de Sentimiento VADER ---
df_comments_reddit_streamlit['vader_neg'] = None
df_comments_reddit_streamlit['vader_neu'] = None
df_comments_reddit_streamlit['vader_pos'] = None
df_comments_reddit_streamlit['vader_compound'] = None


for index, row in df_comments_reddit_streamlit.iterrows():
    comment_text = row['comment_text']
    if isinstance(comment_text, str):
        vs = analyzer_reddit_vader_streamlit.polarity_scores(comment_text)
        df_comments_reddit_streamlit.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = vs.values()
    else:
        df_comments_reddit_streamlit.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = [0.0, 1.0, 0.0, 0.0]


# --- Preparación de Datos para Visualización ---
def categorize_sentiment_streamlit(compound_score):
    if compound_score >= 0.05: return "Positivo"
    elif compound_score <= -0.05: return "Negativo"
    else: return "Neutral"

df_comments_reddit_streamlit['sentiment_category'] = df_comments_reddit_streamlit['vader_compound'].apply(categorize_sentiment_streamlit)
sentiment_counts_streamlit = df_comments_reddit_streamlit['sentiment_category'].value_counts()
average_compound_score_streamlit = df_comments_reddit_streamlit['vader_compound'].mean()

# --- Preparación de Datos para Tendencia Temporal ---
df_comments_reddit_streamlit['comment_datetime'] = pd.to_datetime(df_comments_reddit_streamlit['comment_created_utc'], unit='s', utc=True)
df_comments_reddit_streamlit['comment_date'] = df_comments_reddit_streamlit['comment_datetime'].dt.date
sentiment_over_time_streamlit = df_comments_reddit_streamlit.groupby('comment_date')['vader_compound'].mean()


# --- Visualizaciones en Streamlit ---

# Gráfico de Barras de Categorías de Sentimiento
st.subheader("Distribución de Categorías de Sentimiento")
fig_bar_chart, ax_bar_chart = plt.subplots(figsize=(8, 6))
sns.barplot(x=sentiment_counts_streamlit.index, y=sentiment_counts_streamlit.values, palette="viridis", ax=ax_bar_chart)
ax_bar_chart.set_title('Distribución de Categorías de Sentimiento')
ax_bar_chart.set_xlabel('Categoría de Sentimiento')
ax_bar_chart.set_ylabel('Número de Comentarios')
ax_bar_chart.tick_params(axis='x', rotation=45)
st.pyplot(fig_bar_chart)


# Histograma de Puntuaciones Compuestas de Sentimiento
st.subheader("Distribución de Puntuaciones Compuestas de Sentimiento (VADER)")
fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
sns.histplot(df_comments_reddit_streamlit['vader_compound'], bins=30, kde=True, color='skyblue', ax=ax_hist)
ax_hist.set_title('Distribución de Puntuaciones Compuestas de Sentimiento (VADER)')
ax_hist.set_xlabel('Puntuación Compuesta de Sentimiento (VADER)')
ax_hist.set_ylabel('Frecuencia')
ax_hist.set_xlim(-1, 1)
st.pyplot(fig_hist)

# Gráfico de Líneas de Tendencia Temporal del Sentimiento
st.subheader("Tendencia del Sentimiento Promedio a lo Largo del Tiempo")
ax_line_chart.tick_params(axis='x', rotation=45, horizontalalignment='right')
sns.lineplot(x=sentiment_over_time_streamlit.index, y=sentiment_over_time_streamlit.values, marker='o', ax=ax_line_chart)
ax_line_chart.set_title('Tendencia del Sentimiento Compuesto Promedio a lo Largo del Tiempo')
ax_line_chart.set_xlabel('Fecha del Comentario')
ax_line_chart.set_ylabel('Puntuación Compuesta Promedio de Sentimiento (VADER)')
ax_line_chart.tick_params(axis='x', rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
st.pyplot(fig_line_chart)
