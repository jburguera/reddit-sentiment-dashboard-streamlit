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
st.title("Dashboard de Sentimiento en Reddit: Opinión Pública sobre Tesla en r/TeslaMotors")

# Descripción del Dashboard
st.markdown("""
Dashboard que visualiza el sentimiento público hacia Tesla en el subreddit **r/TeslaMotors**,
utilizando análisis de sentimiento con **VADER** de NLTK.
""")

st.markdown("---")

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
