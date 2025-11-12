import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import praw
import os
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import numpy as np
import re
import base64
from io import BytesIO
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import warnings
import json
import threading
import io

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Download NLTK resources FIRST before importing NLTK modules ---
import nltk
import sys

# Use home directory for NLTK data on Streamlit Cloud
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path at the beginning
nltk.data.path = [nltk_data_dir] + nltk.data.path

# Function to ensure NLTK resource is downloaded
def ensure_nltk_resource(resource_name, resource_path):
    """Download NLTK resource if not already present"""
    try:
        nltk.data.find(resource_path)
        return True
    except LookupError:
        try:
            print(f"Downloading {resource_name}...", file=sys.stderr)
            nltk.download(resource_name, download_dir=nltk_data_dir, quiet=False)
            return True
        except Exception as e:
            print(f"Error downloading {resource_name}: {e}", file=sys.stderr)
            return False

# Download all required resources with verification
# Note: We don't need punkt/punkt_tab anymore since we use simple tokenization
resources_to_download = [
    ('vader_lexicon', 'sentiment/vader_lexicon.zip'),
    ('stopwords', 'corpora/stopwords'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4'),
]

for resource_name, resource_path in resources_to_download:
    ensure_nltk_resource(resource_name, resource_path)

# Now import NLTK modules after resources are downloaded
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# Verify NLTK setup completed
print(f"NLTK data paths: {nltk.data.path[:3]}", file=sys.stderr)

# --- Splash Screen ---
def show_splash_screen():
    # Check if this is the first run of the app
    if 'first_run' not in st.session_state:
        splash_container = st.empty()
        with splash_container.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style="text-align: center; padding: 50px 0;">
                    <h1 style="color: #E31937; font-size: 3rem; margin-bottom: 30px;">Tesla Sentiment Dashboard</h1>
                    <div style="background-color: #E31937; height: 4px; margin: 20px 0;"></div>
                    <p style="font-size: 1.2rem; margin: 20px 0;">Loading Reddit data and initializing analysis...</p>
                    <div class="loader"></div>
                </div>
                <style>
                    .loader {
                        border: 16px solid #f3f3f3;
                        border-radius: 50%;
                        border-top: 16px solid #E31937;
                        width: 120px;
                        height: 120px;
                        animation: spin 2s linear infinite;
                        margin: 30px auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Simulate loading
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.025)  # Adjust for desired splash screen duration
                    progress_bar.progress(i + 1)
                
                st.markdown("""
                <div style="text-align: center; padding: 20px 0;">
                    <p style="font-size: 1.2rem;">Ready!</p>
                </div>
                """, unsafe_allow_html=True)
                time.sleep(0.5)
                
        # Remove splash screen
        splash_container.empty()
        
        # Set flag to avoid showing splash screen again
        st.session_state['first_run'] = False

# --- App Configuration ---
st.set_page_config(
    page_title="Tesla Sentiment Dashboard", 
    page_icon="🚗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show splash screen
show_splash_screen()

# --- Custom CSS ---
st.markdown("""
<style>
    :root {
        --tesla-red: #E31937;
        --tesla-red-dark: #C0392B;
        --positive-color: #1E8449;
        --neutral-color: #707B7C;
        --negative-color: #C0392B;
        --background-light: #ffffff;
        --background-gray: #f9f9f9;
        --text-dark: #2c3e50;
        --text-gray: #666666;
        --text-light: #95a5a6;
        --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.06);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
        --shadow-lg: 0 8px 20px rgba(0, 0, 0, 0.12);
        --transition-smooth: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .main-header {
        font-size: 2.5rem;
        color: var(--tesla-red);
        font-weight: 800;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .card {
        border-radius: 12px;
        padding: 20px;
        background: linear-gradient(145deg, var(--background-light) 0%, var(--background-gray) 100%);
        box-shadow: var(--shadow-md), var(--shadow-sm);
        margin-bottom: 20px;
        transition: var(--transition-smooth);
        border: 1px solid rgba(0, 0, 0, 0.05);
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg), var(--shadow-md);
        border-color: rgba(227, 25, 55, 0.1);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: var(--text-gray);
        text-align: center;
    }
    .tooltip-icon {
        cursor: help;
        color: var(--text-light);
        font-size: 0.9rem;
        margin-left: 5px;
    }
    .metric-card-enhanced {
        position: relative;
        overflow: hidden;
    }
    .metric-card-enhanced::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, var(--tesla-red) 0%, var(--tesla-red-dark) 100%);
    }
    .sticky-toc {
        position: sticky;
        top: 20px;
        background: var(--background-light);
        padding: 15px;
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 20px;
    }
    .sticky-toc h4 {
        margin-top: 0;
        color: var(--tesla-red);
        font-size: 1rem;
    }
    .sticky-toc a {
        display: block;
        padding: 8px 10px;
        color: var(--text-dark);
        text-decoration: none;
        border-radius: 4px;
        transition: var(--transition-smooth);
        font-size: 0.9rem;
    }
    .sticky-toc a:hover {
        background: var(--background-gray);
        color: var(--tesla-red);
        padding-left: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f0f0;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E31937;
        color: white;
    }
    .sentiment-positive {
        color: var(--positive-color);
        font-weight: 600;
    }
    .sentiment-neutral {
        color: var(--neutral-color);
        font-weight: 600;
    }
    .sentiment-negative {
        color: var(--negative-color);
        font-weight: 600;
    }
    .trend-up {
        color: var(--positive-color);
        font-weight: 600;
    }
    .trend-stable {
        color: var(--neutral-color);
        font-weight: 600;
    }
    .trend-down {
        color: var(--negative-color);
        font-weight: 600;
    }
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .elegant-separator {
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, var(--tesla-red) 20%, var(--tesla-red) 80%, transparent 100%);
        border: none;
        margin: 2rem 0;
        opacity: 0.6;
    }
    .insights-panel {
        background: linear-gradient(135deg, var(--tesla-red) 0%, var(--tesla-red-dark) 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 6px 16px rgba(227, 25, 55, 0.25);
    }
    .insights-panel h3 {
        margin-top: 0;
        color: white;
        font-size: 1.4rem;
        margin-bottom: 15px;
    }
    .insight-item {
        background: rgba(255, 255, 255, 0.15);
        padding: 12px 15px;
        border-radius: 8px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        border-left: 4px solid rgba(255, 255, 255, 0.5);
    }
    .footer {
        background: linear-gradient(145deg, var(--text-dark) 0%, #34495e 100%);
        color: #ecf0f1;
        padding: 30px 20px;
        border-radius: 12px;
        margin-top: 40px;
        text-align: center;
        box-shadow: var(--shadow-md);
    }
    .footer a {
        color: var(--tesla-red);
        text-decoration: none;
        transition: var(--transition-smooth);
        font-weight: 500;
    }
    .footer a:hover {
        color: #ff4757;
        text-decoration: underline;
    }
    .footer-links {
        margin: 15px 0;
        font-size: 1.1rem;
    }
    .footer-info {
        margin-top: 15px;
        font-size: 0.9rem;
        color: var(--text-light);
    }
    .download-btn {
        background-color: var(--tesla-red);
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        text-align: center;
        margin-top: 10px;
        transition: var(--transition-smooth);
    }
    .download-btn:hover {
        background-color: var(--tesla-red-dark);
        color: white;
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
    }
    .auto-refresh-active {
        color: #1E8449;
        font-weight: 600;
    }
    .auto-refresh-inactive {
        color: #707B7C;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
def download_link(object_to_download, download_filename, download_link_text):
    """Generate a download link for a DataFrame."""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)
        file_extension = 'csv'
    elif isinstance(object_to_download, dict):
        object_to_download = json.dumps(object_to_download, indent=4)
        file_extension = 'json'
    else:
        object_to_download = str(object_to_download)
        file_extension = 'txt'
    
    # Create base64 encoded string for download
    b64 = base64.b64encode(object_to_download.encode()).decode()
    href = f'<a href="data:file/{file_extension};base64,{b64}" download="{download_filename}" class="download-btn">{download_link_text}</a>'
    return href

def detect_sentiment_trend(df_comments, window_size=3):
    """Detect trends in sentiment over time."""
    # Ensure we have dates in ascending order
    df_trend = df_comments.groupby('comment_date')['vader_compound'].mean().reset_index()
    df_trend = df_trend.sort_values('comment_date')
    
    if len(df_trend) < window_size:
        return "Insufficient data for trend analysis"
    
    # Calculate rolling average for smoothing
    df_trend['rolling_avg'] = df_trend['vader_compound'].rolling(window=window_size, min_periods=1).mean()
    
    # Calculate percentage change
    df_trend['pct_change'] = df_trend['rolling_avg'].pct_change() * 100
    
    # Get the latest percentage change
    latest_pct_change = df_trend['pct_change'].iloc[-1]
    
    # Determine trend based on percentage change
    if pd.isna(latest_pct_change):
        trend = "Stable"
        description = "Sentiment trend appears stable."
    elif latest_pct_change > 5:
        trend = "Up"
        description = f"Sentiment is trending upward ({latest_pct_change:.1f}% increase)."
    elif latest_pct_change < -5:
        trend = "Down"
        description = f"Sentiment is trending downward ({abs(latest_pct_change):.1f}% decrease)."
    else:
        trend = "Stable"
        description = "Sentiment trend appears stable."
    
    # Check absolute value
    latest_value = df_trend['rolling_avg'].iloc[-1]
    absolute_desc = ""
    if latest_value > 0.2:
        absolute_desc = "Overall sentiment is strongly positive."
    elif latest_value > 0.05:
        absolute_desc = "Overall sentiment is positive."
    elif latest_value < -0.2:
        absolute_desc = "Overall sentiment is strongly negative."
    elif latest_value < -0.05:
        absolute_desc = "Overall sentiment is negative."
    else:
        absolute_desc = "Overall sentiment is neutral."
    
    # Combine trend and absolute descriptions
    full_description = f"{description} {absolute_desc}"
    
    return {
        "trend": trend,
        "description": full_description,
        "latest_value": latest_value,
        "latest_change": latest_pct_change if not pd.isna(latest_pct_change) else 0
    }

# --- NLP Functions ---
def preprocess_text(text, custom_stopwords=None):
    """Preprocess text for NLP tasks."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs, Reddit formatting, special chars, and numbers
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove Reddit links [text](url)
    text = re.sub(r'\*\*|\*', '', text)  # Remove Reddit bold/italic
    text = re.sub(r'&amp;', '&', text)  # Replace HTML entities
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces

    # Tokenize - use simple split() instead of NLTK word_tokenize
    # This avoids punkt/punkt_tab dependency issues on Streamlit Cloud
    # Split on whitespace and filter empty strings
    tokens = [token for token in text.split() if token]
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    if custom_stopwords:
        stop_words.update(custom_stopwords)
    
    # Add Reddit-specific and Tesla-specific stopwords
    reddit_stopwords = {'edit', 'deleted', 'removed', 'submission', 'comment', 'post', 'reddit', 'subreddit'}
    tesla_stopwords = {'tesla', 'model', 'car', 'ev', 'vehicle', 'drive', 'driving', 'drove', 'elon', 'musk'}
    stop_words.update(reddit_stopwords)
    stop_words.update(tesla_stopwords)
    
    # Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens

def create_wordcloud(text_data, sentiment_type, colormap):
    """Create a word cloud from text data."""
    # Join all tokens into a single string
    if isinstance(text_data, list) and len(text_data) > 0 and isinstance(text_data[0], list):
        # If text_data is a list of token lists, flatten it
        all_tokens = [token for token_list in text_data for token in token_list]
        text = ' '.join(all_tokens)
    elif isinstance(text_data, list):
        text = ' '.join(text_data)
    else:
        text = str(text_data)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        colormap=colormap,
        contour_width=1,
        contour_color='gray',
        collocations=False
    ).generate(text)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f"{sentiment_type} Sentiment Word Cloud", fontsize=16)
    ax.axis('off')
    
    return fig

def build_lda_model(documents, num_topics=5):
    """Build an LDA topic model from preprocessed documents."""
    # Create dictionary
    dictionary = corpora.Dictionary(documents)
    
    # Filter out extreme values (words appearing in fewer than 5 docs or more than 50% of docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    
    # Create document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in documents]
    
    # Build LDA model
    lda_model = gensim.models.LdaModel(
        corpus=doc_term_matrix,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=10,
        alpha='auto',
        eta='auto'
    )
    
    # Calculate coherence score
    coherence_model = CoherenceModel(
        model=lda_model, 
        texts=documents, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence_score = coherence_model.get_coherence()
    
    return {
        'model': lda_model,
        'dictionary': dictionary,
        'doc_term_matrix': doc_term_matrix,
        'coherence_score': coherence_score
    }

def get_topic_visualizations(lda_results, top_n_words=10):
    """Get visualizations for LDA topic model results."""
    topics = []
    model = lda_results['model']
    
    # Get the top n words for each topic
    for idx, topic in model.print_topics(num_words=top_n_words):
        topic_terms = topic.split('+')
        topic_words = []
        for term in topic_terms:
            # Extract just the word, removing the weight and formatting
            word = term.strip().split('*')[1].strip().replace('"', '').replace("'", '')
            topic_words.append(word)
        
        topics.append({
            'id': idx,
            'words': topic_words
        })
    
    # Create pyLDAvis visualization
    vis_data = pyLDAvis.gensim_models.prepare(
        model, 
        lda_results['doc_term_matrix'], 
        lda_results['dictionary'],
        sort_topics=False
    )
    
    # Convert visualization to HTML string
    vis_html = pyLDAvis.prepared_data_to_html(vis_data)
    
    return {
        'topics': topics,
        'html': vis_html,
        'coherence_score': lda_results['coherence_score']
    }

# --- Reddit Data Functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_reddit_credentials():
    """Load Reddit API credentials from file."""
    try:
        credentials = {}
        with open("reddit_credentials.txt", "r") as f:
            for line in f:
                if '=' in line:
                    key, value = [k.strip() for k in line.split('=', 1)]
                    credentials[key] = value
        
        required_keys = ["CLIENT_ID", "CLIENT_SECRET", "USER_AGENT"]
        for key in required_keys:
            if key not in credentials:
                raise KeyError(f"Missing {key} in credentials file")
        
        return credentials
    except FileNotFoundError:
        st.error("Credentials file not found. Please create 'reddit_credentials.txt' in the app directory.")
        return None
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return None

def fetch_reddit_data(credentials, subreddit_name, post_limit, min_comments, time_filter):
    """Fetch data from Reddit API with progress tracking."""
    try:
        # Initialize Reddit API
        reddit = praw.Reddit(
            client_id=credentials["CLIENT_ID"],
            client_secret=credentials["CLIENT_SECRET"],
            user_agent=credentials["USER_AGENT"]
        )
        
        subreddit = reddit.subreddit(subreddit_name)
        
        # Create progress bar with status
        progress_text = "🔍 Conectando con Reddit..."
        progress_bar = st.progress(0, text=progress_text)
        status_text = st.empty()

        # Get hot posts
        status_text.info(f"📥 Descargando posts de r/{subreddit_name}...")
        hot_posts = list(subreddit.hot(limit=post_limit))
        filtered_posts = [post for post in hot_posts if post.num_comments >= min_comments]

        # Check if we have posts
        if not filtered_posts:
            st.warning(f"No posts found with at least {min_comments} comments. Try adjusting your filters.")
            status_text.empty()
            return None, None

        status_text.success(f"✅ Encontrados {len(filtered_posts)} posts con ≥{min_comments} comentarios")

        # Prepare for data collection
        post_data_list = []
        comment_data_list = []
        total_comments_collected = 0

        # Process posts with progress tracking
        total_posts = len(filtered_posts)
        for i, post in enumerate(filtered_posts):
            # Update progress
            progress = int((i + 1) / total_posts * 100)
            progress_bar.progress(
                progress,
                text=f"📝 Procesando post {i+1}/{total_posts} • {total_comments_collected} comentarios recopilados"
            )
            
            # Extract post data
            post_data = {
                "post_id": post.id,
                "post_title": post.title,
                "post_text": post.selftext,
                "post_url": f"https://www.reddit.com{post.permalink}",
                "post_score": post.score,
                "post_upvote_ratio": post.upvote_ratio,
                "post_num_comments": post.num_comments,
                "post_created_utc": post.created_utc,
                "post_author": str(post.author)
            }
            post_data_list.append(post_data)
            
            # Extract comment data
            post.comments.replace_more(limit=0)  # Only get top-level comments
            for comment in post.comments:
                comment_data = {
                    "comment_id": comment.id,
                    "comment_text": comment.body,
                    "comment_author": str(comment.author),
                    "comment_score": comment.score,
                    "comment_created_utc": comment.created_utc,
                    "post_id": post.id,
                    "post_title": post.title
                }
                comment_data_list.append(comment_data)
                total_comments_collected += 1

        # Clear progress bar and show final status
        progress_bar.empty()
        status_text.success(f"✅ Recopilación completa: {len(filtered_posts)} posts • {total_comments_collected} comentarios")

        # Create DataFrames
        df_posts = pd.DataFrame(post_data_list)
        df_comments = pd.DataFrame(comment_data_list)

        return df_posts, df_comments
    
    except Exception as e:
        st.error(f"Error fetching data from Reddit: {e}")
        return None, None

def analyze_sentiment(df_comments):
    """Perform sentiment analysis on comments."""
    analyzer = SentimentIntensityAnalyzer()

    # Create progress bar with status
    progress_bar = st.progress(0, text="🔬 Iniciando análisis de sentiment...")
    status_text = st.empty()

    # Prepare columns for sentiment scores
    df_comments['vader_neg'] = None
    df_comments['vader_neu'] = None
    df_comments['vader_pos'] = None
    df_comments['vader_compound'] = None

    # Process comments with progress tracking
    total_comments = len(df_comments)
    status_text.info(f"🔬 Paso 1/3: Analizando sentiment con VADER en {total_comments} comentarios...")

    for i, (index, row) in enumerate(df_comments.iterrows()):
        # Update progress every 10 comments to improve performance
        if i % 10 == 0 or i == total_comments - 1:
            progress = int((i + 1) / total_comments * 33)  # First 33% for sentiment analysis
            progress_bar.progress(
                progress,
                text=f"🔬 Analizando sentiment: {i+1}/{total_comments} ({(i+1)/total_comments*100:.1f}%)"
            )

        comment_text = row['comment_text']
        if isinstance(comment_text, str):
            vs = analyzer.polarity_scores(comment_text)
            df_comments.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = vs.values()
        else:
            df_comments.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = [0.0, 1.0, 0.0, 0.0]
    
    # Categorize sentiment
    progress_bar.progress(33, text="📊 Paso 2/3: Categorizando sentiment (positivo/neutral/negativo)...")
    status_text.info("📊 Paso 2/3: Categorizando sentiment...")

    df_comments['sentiment_category'] = df_comments['vader_compound'].apply(
        lambda score: "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
    )

    # Add datetime columns
    df_comments['comment_datetime'] = pd.to_datetime(df_comments['comment_created_utc'], unit='s', utc=True)
    df_comments['comment_date'] = df_comments['comment_datetime'].dt.date
    df_comments['comment_hour'] = df_comments['comment_datetime'].dt.hour

    progress_bar.progress(50, text="⏰ Procesando datos temporales...")

    # Preprocess comment text for NLP analysis
    status_text.info(f"🔤 Paso 3/3: Preprocesando texto para NLP en {total_comments} comentarios...")
    df_comments['tokens'] = None

    for i, (index, row) in enumerate(df_comments.iterrows()):
        if i % 20 == 0 or i == total_comments - 1:
            progress = 50 + int((i + 1) / total_comments * 50)  # 50-100% for preprocessing
            progress_bar.progress(
                progress,
                text=f"🔤 Tokenizando: {i+1}/{total_comments} ({(i+1)/total_comments*100:.1f}%)"
            )

        tokens = preprocess_text(row['comment_text'])
        df_comments.loc[index, 'tokens'] = str(tokens)  # Store as string for serialization

    # Get sentiment distribution
    sentiment_counts = df_comments['sentiment_category'].value_counts()
    pos_count = sentiment_counts.get('Positive', 0)
    neu_count = sentiment_counts.get('Neutral', 0)
    neg_count = sentiment_counts.get('Negative', 0)

    # Show final summary
    progress_bar.empty()
    status_text.success(
        f"✅ Análisis completado: {pos_count} positivos • {neu_count} neutrales • {neg_count} negativos"
    )

    return df_comments

def create_sentiment_summary(df_comments):
    """Create summary metrics for sentiment analysis."""
    total_comments = len(df_comments)
    
    if total_comments == 0:
        return {
            "total_comments": 0,
            "avg_compound": 0,
            "positive_pct": 0,
            "neutral_pct": 0,
            "negative_pct": 0
        }
    
    sentiment_counts = df_comments['sentiment_category'].value_counts()
    
    positive_count = sentiment_counts.get("Positive", 0)
    neutral_count = sentiment_counts.get("Neutral", 0)
    negative_count = sentiment_counts.get("Negative", 0)
    
    summary = {
        "total_comments": total_comments,
        "avg_compound": df_comments['vader_compound'].mean(),
        "positive_pct": (positive_count / total_comments) * 100,
        "neutral_pct": (neutral_count / total_comments) * 100,
        "negative_pct": (negative_count / total_comments) * 100
    }
    
    return summary

# --- Automatic Refresh Function ---
def auto_refresh_data():
    """Function to automatically refresh data at specified intervals."""
    while st.session_state.get('auto_refresh_enabled', False):
        # Sleep for the specified interval
        time.sleep(st.session_state.get('refresh_interval', 3600))
        
        # Set refresh flag to trigger data reload
        st.session_state['trigger_refresh'] = True
        
        # Update last refresh time
        st.session_state['last_refresh'] = datetime.now()
        
        # Force a rerun of the app
        st.rerun()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## Configuración")
    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Reddit Configuration
    st.markdown("### Datos de Reddit")
    subreddit_name = st.text_input("Subreddit", "TeslaMotors", help="Nombre del subreddit (sin r/)")

    post_limit = st.slider(
        "Número de posts",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Más posts significa más datos pero carga más lenta"
    )

    min_comments = st.slider(
        "Mínimo de comentarios por post",
        min_value=10,
        max_value=300,
        value=150,
        step=10,
        help="Filtrar posts por número mínimo de comentarios"
    )

    time_filter = st.selectbox(
        "Período de tiempo",
        ["day", "week", "month", "year", "all"],
        index=1,
        help="Time period for hot posts"
    )
    
    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Visualization Settings
    st.markdown("### Visualización")

    color_theme = st.selectbox(
        "Tema de color",
        ["Tesla", "Viridis", "Plasma", "Inferno", "Magma"],
        index=0
    )

    chart_style = st.selectbox(
        "Estilo de gráficos",
        ["whitegrid", "darkgrid", "white", "dark", "ticks"],
        index=0
    )

    num_topics = st.slider(
        "Número de tópicos",
        min_value=2,
        max_value=10,
        value=5,
        help="Más tópicos muestran distinciones más finas pero pueden ser más difíciles de interpretar"
    )
    
    if color_theme == "Tesla":
        color_positive = "#1E8449"  # Green
        color_neutral = "#707B7C"   # Gray
        color_negative = "#C0392B"  # Red
        color_main = "#E31937"      # Tesla Red
        wordcloud_positive_cmap = "Greens"
        wordcloud_neutral_cmap = "Greys"
        wordcloud_negative_cmap = "Reds"
    else:
        color_positive = None
        color_neutral = None
        color_negative = None
        color_main = None
        wordcloud_positive_cmap = color_theme.lower()
        wordcloud_neutral_cmap = "Greys"
        wordcloud_negative_cmap = color_theme.lower()
    
    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Auto-refresh settings
    st.markdown("### Actualización automática")

    # Initialize auto-refresh state if not exists
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state['auto_refresh_enabled'] = False

    auto_refresh = st.checkbox(
        "Activar actualización automática",
        value=st.session_state.get('auto_refresh_enabled', False),
        help="Actualizar datos automáticamente en intervalos específicos"
    )

    refresh_interval_options = {
        "30 minutos": 1800,
        "1 hora": 3600,
        "3 horas": 10800,
        "6 horas": 21600,
        "12 horas": 43200,
        "24 horas": 86400
    }

    refresh_interval_key = st.selectbox(
        "Intervalo de actualización",
        list(refresh_interval_options.keys()),
        index=1,  # Default to 1 hour
        disabled=not auto_refresh
    )
    
    # Update session state
    current_refresh_state = st.session_state.get('auto_refresh_enabled', False)
    if auto_refresh != current_refresh_state:
        st.session_state['auto_refresh_enabled'] = auto_refresh
        st.session_state['refresh_interval'] = refresh_interval_options[refresh_interval_key]
        
        # Start or stop auto-refresh thread
        if auto_refresh:
            threading.Thread(target=auto_refresh_data, daemon=True).start()
    
    # Show auto-refresh status
    if auto_refresh:
        next_refresh = st.session_state.get('last_refresh', datetime.now()) + timedelta(seconds=refresh_interval_options[refresh_interval_key])
        st.markdown(f"""
        <p class="auto-refresh-active">
            Auto-refresh active. Next refresh at: {next_refresh.strftime('%H:%M:%S')}
        </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <p class="auto-refresh-inactive">
            Auto-refresh inactive
        </p>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Search/Analyze button - main action
    st.markdown("### Ready to Analyze?")
    analyze_button = st.button("🔍 Analyze Sentiment", type="primary", width='stretch')

    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Manual refresh button (only show if data has been loaded before)
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        refresh_data = st.button("🔄 Refresh Data", key="refresh_data", width='stretch')
    else:
        refresh_data = False

    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Table of Contents (only show when data is loaded)
    if 'data_loaded' in st.session_state and st.session_state['data_loaded']:
        st.markdown("""
        <div class="sticky-toc">
            <h4>📑 Tabla de Contenidos</h4>
            <a href="#sentiment-distribution">📊 Distribución</a>
            <a href="#temporal-analysis">⏱️ Análisis Temporal</a>
            <a href="#top-posts">⭐ Posts Destacados</a>
            <a href="#word-analysis">💬 Análisis de Palabras</a>
            <a href="#data-export">📥 Exportar Datos</a>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    st.markdown("#### About")
    st.markdown("This dashboard analyzes sentiment in Reddit posts about Tesla from the selected subreddit.")
    st.markdown("Configure the parameters above and click '🔍 Analyze Sentiment' to start.")

# --- Main Content ---
st.markdown('<p class="main-header">Dashboard de sentiment Tesla</p>', unsafe_allow_html=True)

st.markdown("""
Analiza el sentiment del público sobre Tesla en Reddit usando análisis VADER (NLTK)
para clasificar comentarios como positivos, negativos o neutrales.
""")

# --- Main Process ---
# Load Reddit credentials
credentials = load_reddit_credentials()
if not credentials:
    st.error("Failed to load Reddit credentials. Please check your credentials file.")
    st.stop()

# Set seaborn style
sns.set_style(chart_style)

# Fetch data with caching
@st.cache_data(ttl=900, show_spinner=False)
def get_analyzed_data(credentials, subreddit, limit, min_comments, time_filter):
    """Fetch and analyze Reddit data with caching"""
    df_posts, df_comments = fetch_reddit_data(
        credentials, subreddit, limit, min_comments, time_filter
    )

    if df_posts is None or df_comments is None or len(df_comments) == 0:
        return None, None

    df_comments_analyzed = analyze_sentiment(df_comments)
    return df_posts, df_comments_analyzed

# Check if analyze button was pressed or if we should refresh data
if analyze_button or refresh_data or ('data_loaded' in st.session_state and st.session_state['data_loaded']):
    # Set flag that data has been loaded
    st.session_state['data_loaded'] = True

    # Check if we should refresh data
    if 'last_refresh' not in st.session_state or refresh_data or analyze_button:
        st.session_state['last_refresh'] = datetime.now()
        df_posts, df_comments = get_analyzed_data(
            credentials, subreddit_name, post_limit, min_comments, time_filter
        )
    else:
        df_posts, df_comments = get_analyzed_data(
            credentials, subreddit_name, post_limit, min_comments, time_filter
        )

    # Continue only if we have data
    if df_posts is None or df_comments is None or len(df_comments) == 0:
        st.error("No data available. Please adjust your parameters and try again.")
        st.stop()

    # Calculate sentiment trend
    sentiment_trend_info = detect_sentiment_trend(df_comments)

    # --- Create summary metrics ---
    sentiment_summary = create_sentiment_summary(df_comments)

    # Display summary metrics with trend indicators and tooltips
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown('<div class="card metric-card-enhanced">', unsafe_allow_html=True)
        st.metric(
            label="💬 Total Comments",
            value=f"{sentiment_summary['total_comments']:,}",
            help="Número total de comentarios analizados de los posts seleccionados"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        sentiment_score = sentiment_summary["avg_compound"]
        sentiment_label = "Positivo" if sentiment_score >= 0.05 else ("Negativo" if sentiment_score <= -0.05 else "Neutral")
        st.markdown('<div class="card metric-card-enhanced">', unsafe_allow_html=True)
        st.metric(
            label="📊 Sentiment Score",
            value=f"{sentiment_score:.3f}",
            delta=sentiment_label,
            help="Score VADER promedio (-1 = muy negativo, +1 = muy positivo). Threshold: ≥0.05 positivo, ≤-0.05 negativo"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card metric-card-enhanced">', unsafe_allow_html=True)
        st.metric(
            label="😊 Comentarios Positivos",
            value=f"{sentiment_summary['positive_pct']:.1f}%",
            delta=f"{int(sentiment_summary['positive_pct'] * sentiment_summary['total_comments'] / 100)} comentarios",
            help="Porcentaje de comentarios con score VADER ≥0.05 (sentiment positivo)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card metric-card-enhanced">', unsafe_allow_html=True)
        st.metric(
            label="😞 Comentarios Negativos",
            value=f"{sentiment_summary['negative_pct']:.1f}%",
            delta=f"{int(sentiment_summary['negative_pct'] * sentiment_summary['total_comments'] / 100)} comentarios",
            delta_color="inverse",
            help="Porcentaje de comentarios con score VADER ≤-0.05 (sentiment negativo)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col5:
        st.markdown('<div class="card metric-card-enhanced">', unsafe_allow_html=True)
        if isinstance(sentiment_trend_info, dict):
            trend_emoji = "📈" if sentiment_trend_info['trend'] == "Up" else ("📉" if sentiment_trend_info['trend'] == "Down" else "➡️")
            st.metric(
                label=f"{trend_emoji} Tendencia",
                value=sentiment_trend_info["trend"],
                delta=f"{sentiment_trend_info['latest_change']:.1f}%",
                help="Tendencia de sentiment basada en cambio porcentual con promedio móvil de 3 días"
            )
        else:
            st.metric(
                label="📊 Tendencia",
                value="N/A",
                help="Insuficientes datos para calcular tendencia"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show trend insight
    if isinstance(sentiment_trend_info, dict):
        st.info(f"📊 Trend Insight: {sentiment_trend_info['description']}")

    # --- Key Insights Panel ---
    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Calculate key insights
    most_active_hour = df_comments.groupby('comment_hour').size().idxmax()
    most_active_day = df_comments.groupby('comment_date').size().idxmax()
    avg_comments_per_post = len(df_comments) / len(df_posts) if len(df_posts) > 0 else 0

    # Find dominant sentiment
    sentiment_counts = df_comments['sentiment_category'].value_counts()
    dominant_sentiment = sentiment_counts.idxmax()
    dominant_sentiment_pct = (sentiment_counts.max() / len(df_comments)) * 100

    # Sentiment comparison
    positive_pct = sentiment_summary['positive_pct']
    negative_pct = sentiment_summary['negative_pct']
    sentiment_balance = positive_pct - negative_pct

    st.markdown("""
    <div class="insights-panel">
        <h3>🔍 Key Insights del Análisis</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid #E31937;">
            <h4 style="color: #E31937; margin-top: 0;">💬 Engagement</h4>
            <p style="font-size: 0.95rem; margin: 8px 0;">
                <strong style="font-size: 1.8rem; display: block; margin: 10px 0;">{:.1f}</strong>
                comentarios promedio por post
            </p>
            <p style="font-size: 0.85rem; color: #666; margin-top: 10px;">
                Hora más activa: <strong>{}:00 UTC</strong>
            </p>
        </div>
        """.format(avg_comments_per_post, most_active_hour), unsafe_allow_html=True)

    with col2:
        balance_color = "#1E8449" if sentiment_balance > 0 else ("#C0392B" if sentiment_balance < 0 else "#707B7C")
        balance_icon = "📈" if sentiment_balance > 0 else ("📉" if sentiment_balance < 0 else "➡️")
        balance_text = "más positivo" if sentiment_balance > 0 else ("más negativo" if sentiment_balance < 0 else "equilibrado")

        st.markdown("""
        <div class="card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid {};">
            <h4 style="color: {}; margin-top: 0;">{} Balance de Sentiment</h4>
            <p style="font-size: 0.95rem; margin: 8px 0;">
                <strong style="font-size: 1.8rem; display: block; margin: 10px 0; color: {};">{:+.1f}%</strong>
                sentiment {}
            </p>
            <p style="font-size: 0.85rem; color: #666; margin-top: 10px;">
                {:.1f}% positivo vs {:.1f}% negativo
            </p>
        </div>
        """.format(
            balance_color, balance_color, balance_icon, balance_color,
            sentiment_balance, balance_text, positive_pct, negative_pct
        ), unsafe_allow_html=True)

    with col3:
        dominant_color = "#1E8449" if dominant_sentiment == "Positive" else ("#C0392B" if dominant_sentiment == "Negative" else "#707B7C")
        dominant_icon = "😊" if dominant_sentiment == "Positive" else ("😞" if dominant_sentiment == "Negative" else "😐")

        st.markdown("""
        <div class="card" style="background: rgba(255, 255, 255, 0.95); border-left: 4px solid {};">
            <h4 style="color: {}; margin-top: 0;">{} Sentiment Dominante</h4>
            <p style="font-size: 0.95rem; margin: 8px 0;">
                <strong style="font-size: 1.8rem; display: block; margin: 10px 0; color: {};">{}</strong>
                representa el {:.1f}% de comentarios
            </p>
            <p style="font-size: 0.85rem; color: #666; margin-top: 10px;">
                Score promedio: <strong>{:.3f}</strong>
            </p>
        </div>
        """.format(
            dominant_color, dominant_color, dominant_icon, dominant_color,
            dominant_sentiment, dominant_sentiment_pct, sentiment_summary['avg_compound']
        ), unsafe_allow_html=True)

    # --- Sparklines for Quick Trends ---
    st.markdown("### 📈 Tendencias Rápidas")

    spark_col1, spark_col2, spark_col3 = st.columns(3)

    # Sentiment over time sparkline
    with spark_col1:
        sentiment_by_date = df_comments.groupby('comment_date')['vader_compound'].mean().reset_index()
        sentiment_by_date = sentiment_by_date.sort_values('comment_date')

        fig_spark1 = go.Figure()
        fig_spark1.add_trace(go.Scatter(
            x=sentiment_by_date['comment_date'],
            y=sentiment_by_date['vader_compound'],
            mode='lines',
            line=dict(color='#E31937', width=2),
            fill='tozeroy',
            fillcolor='rgba(227, 25, 55, 0.1)'
        ))
        fig_spark1.update_layout(
            title="Sentiment en el Tiempo",
            height=150,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=True, showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_spark1, width='stretch')

    # Comments volume sparkline
    with spark_col2:
        comments_by_date = df_comments.groupby('comment_date').size().reset_index()
        comments_by_date.columns = ['comment_date', 'count']
        comments_by_date = comments_by_date.sort_values('comment_date')

        fig_spark2 = go.Figure()
        fig_spark2.add_trace(go.Bar(
            x=comments_by_date['comment_date'],
            y=comments_by_date['count'],
            marker=dict(color='#1E8449'),
            opacity=0.7
        ))
        fig_spark2.update_layout(
            title="Volumen de Comentarios",
            height=150,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=True, showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_spark2, width='stretch')

    # Sentiment distribution sparkline
    with spark_col3:
        sentiment_dist = df_comments['sentiment_category'].value_counts()
        colors = {'Positive': '#1E8449', 'Neutral': '#707B7C', 'Negative': '#C0392B'}

        fig_spark3 = go.Figure()
        fig_spark3.add_trace(go.Bar(
            x=sentiment_dist.index,
            y=sentiment_dist.values,
            marker=dict(color=[colors[cat] for cat in sentiment_dist.index]),
            opacity=0.8
        ))
        fig_spark3.update_layout(
            title="Distribución de Sentiment",
            height=150,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(showticklabels=True),
            yaxis=dict(showticklabels=True, showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_spark3, width='stretch')

    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # --- Visualization Tabs ---
    tabs = st.tabs(["Sentiment Distribution", "Temporal Analysis", "Top Posts", "Word Analysis", "Data Export & Insights"])
    
    # Tab 1: Sentiment Distribution
    with tabs[0]:
        st.markdown('<p class="subheader">Sentiment Distribution Analysis</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution pie chart
            sentiment_counts = df_comments['sentiment_category'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            
            colors = {
                'Positive': color_positive or '#1E8449',
                'Neutral': color_neutral or '#707B7C',
                'Negative': color_negative or '#C0392B'
            }
            
            fig_pie = px.pie(
                sentiment_counts, 
                values='Count', 
                names='Sentiment',
                color='Sentiment',
                color_discrete_map=colors,
                hole=0.4,
                title="Distribution of Sentiment Categories"
            )
            
            fig_pie.update_layout(
                legend_title="Sentiment",
                font=dict(size=12),
                title_font=dict(size=16),
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            # Sentiment score histogram
            fig_hist = px.histogram(
                df_comments, 
                x='vader_compound',
                nbins=30,
                title="Distribution of Sentiment Scores",
                color_discrete_sequence=[color_main or '#E31937']
            )
            
            fig_hist.add_vline(x=0.05, line_dash="dash", line_color="green", annotation_text="Positive threshold")
            fig_hist.add_vline(x=-0.05, line_dash="dash", line_color="red", annotation_text="Negative threshold")
            
            fig_hist.update_layout(
                xaxis_title="Sentiment Score",
                yaxis_title="Number of Comments",
                font=dict(size=12),
                title_font=dict(size=16),
                margin=dict(t=50, b=20, l=20, r=20)
            )
            
            st.plotly_chart(fig_hist, width='stretch')
        
        # Combined violin and box plot
        st.markdown("### Sentiment Score Distribution by Category")
        
        fig_violin = px.violin(
            df_comments,
            x="sentiment_category",
            y="vader_compound",
            color="sentiment_category",
            color_discrete_map=colors,
            box=True,
            points="all",
            title="Sentiment Score Distribution by Category"
        )
        
        fig_violin.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Sentiment Score",
            font=dict(size=12),
            title_font=dict(size=16)
        )
        
        st.plotly_chart(fig_violin, width='stretch')
    
    # Tab 2: Temporal Analysis
    with tabs[1]:
        st.markdown('<p class="subheader">Sentiment Over Time</p>', unsafe_allow_html=True)
        
        # Group data by date
        sentiment_by_date = df_comments.groupby('comment_date')['vader_compound'].mean().reset_index()
        sentiment_by_date.columns = ['Date', 'Average Sentiment']
        
        # Line chart for sentiment over time
        fig_line = px.line(
            sentiment_by_date,
            x='Date',
            y='Average Sentiment',
            markers=True,
            title="Average Sentiment Score Over Time",
            color_discrete_sequence=[color_main or '#E31937']
        )
        
        fig_line.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="Positive threshold")
        fig_line.add_hline(y=-0.05, line_dash="dash", line_color="red", annotation_text="Negative threshold")
        
        fig_line.update_layout(
            xaxis_title="Date",
            yaxis_title="Average Sentiment Score",
            font=dict(size=12),
            title_font=dict(size=16)
        )
        
        st.plotly_chart(fig_line, width='stretch')
        
        # Group data by hour
        sentiment_by_hour = df_comments.groupby('comment_hour')['vader_compound'].mean().reset_index()
        sentiment_by_hour.columns = ['Hour of Day', 'Average Sentiment']
        
        # Bar chart for sentiment by hour
        fig_hour = px.bar(
            sentiment_by_hour,
            x='Hour of Day',
            y='Average Sentiment',
            title="Average Sentiment Score by Hour of Day",
            color='Average Sentiment',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        fig_hour.update_layout(
            xaxis_title="Hour of Day (UTC)",
            yaxis_title="Average Sentiment Score",
            font=dict(size=12),
            title_font=dict(size=16),
            coloraxis_colorbar=dict(title="Sentiment")
        )
        
        st.plotly_chart(fig_hour, width='stretch')
        
        # Volume of comments over time
        comment_volume = df_comments.groupby('comment_date').size().reset_index()
        comment_volume.columns = ['Date', 'Number of Comments']
        
        fig_volume = px.bar(
            comment_volume,
            x='Date',
            y='Number of Comments',
            title="Volume of Comments Over Time",
            color_discrete_sequence=[color_main or '#E31937']
        )
        
        fig_volume.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Comments",
            font=dict(size=12),
            title_font=dict(size=16)
        )
        
        st.plotly_chart(fig_volume, width='stretch')
    
    # Tab 3: Top Posts
    with tabs[2]:
        st.markdown('<p class="subheader">Top Posts Analysis</p>', unsafe_allow_html=True)
        
        # Calculate average sentiment per post
        post_sentiment = df_comments.groupby('post_id').agg({
            'vader_compound': 'mean',
            'post_title': 'first',
            'comment_id': 'count'
        }).reset_index()
        
        post_sentiment.columns = ['Post ID', 'Avg Sentiment', 'Post Title', 'Comment Count']
        
        # Join with post data to get more metrics
        post_sentiment = post_sentiment.merge(
            df_posts[['post_id', 'post_score', 'post_upvote_ratio', 'post_url']],
            left_on='Post ID',
            right_on='post_id'
        )
        
        post_sentiment['sentiment_category'] = post_sentiment['Avg Sentiment'].apply(
            lambda score: "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
        )
        
        # Most positive posts
        st.markdown("### Most Positive Posts")
        positive_posts = post_sentiment.sort_values('Avg Sentiment', ascending=False).head(5)
        
        for i, row in positive_posts.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h4>{row['Post Title']}</h4>
                    <p><span class="sentiment-positive">Sentiment Score: {row['Avg Sentiment']:.2f}</span> | 
                       Upvotes: {row['post_score']} | Comments: {row['Comment Count']}</p>
                    <p><a href="{row['post_url']}" target="_blank">View on Reddit</a></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Most negative posts
        st.markdown("### Most Negative Posts")
        negative_posts = post_sentiment.sort_values('Avg Sentiment', ascending=True).head(5)
        
        for i, row in negative_posts.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h4>{row['Post Title']}</h4>
                    <p><span class="sentiment-negative">Sentiment Score: {row['Avg Sentiment']:.2f}</span> | 
                       Upvotes: {row['post_score']} | Comments: {row['Comment Count']}</p>
                    <p><a href="{row['post_url']}" target="_blank">View on Reddit</a></p>
                </div>
                """, unsafe_allow_html=True)
        
        # Most discussed posts
        st.markdown("### Most Discussed Posts")
        discussed_posts = post_sentiment.sort_values('Comment Count', ascending=False).head(5)
        
        for i, row in discussed_posts.iterrows():
            sentiment_class = "sentiment-positive" if row['Avg Sentiment'] >= 0.05 else ("sentiment-negative" if row['Avg Sentiment'] <= -0.05 else "sentiment-neutral")
            
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <h4>{row['Post Title']}</h4>
                    <p><span class="{sentiment_class}">Sentiment Score: {row['Avg Sentiment']:.2f}</span> | 
                       Upvotes: {row['post_score']} | Comments: {row['Comment Count']}</p>
                    <p><a href="{row['post_url']}" target="_blank">View on Reddit</a></p>
                </div>
                """, unsafe_allow_html=True)
    
    # Tab 4: Word Analysis
    with tabs[3]:
        st.markdown('<p class="subheader">Word Analysis</p>', unsafe_allow_html=True)
    
        # Prepare token data for analysis
        df_comments['tokens_list'] = df_comments['tokens'].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    
        # Word Clouds by Sentiment
        st.markdown("### Word Clouds by Sentiment Category")
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("#### Positive Comments")
            positive_comments = df_comments[df_comments['sentiment_category'] == 'Positive']
            if len(positive_comments) > 0:
                positive_tokens = positive_comments['tokens_list'].tolist()
                try:
                    fig_positive = create_wordcloud(positive_tokens, "Positive", wordcloud_positive_cmap)
                    st.pyplot(fig_positive)
                    plt.close(fig_positive)
                except Exception as e:
                    st.warning(f"Not enough positive words to generate word cloud: {str(e)}")
            else:
                st.info("No positive comments to analyze")
    
        with col2:
            st.markdown("#### Neutral Comments")
            neutral_comments = df_comments[df_comments['sentiment_category'] == 'Neutral']
            if len(neutral_comments) > 0:
                neutral_tokens = neutral_comments['tokens_list'].tolist()
                try:
                    fig_neutral = create_wordcloud(neutral_tokens, "Neutral", wordcloud_neutral_cmap)
                    st.pyplot(fig_neutral)
                    plt.close(fig_neutral)
                except Exception as e:
                    st.warning(f"Not enough neutral words to generate word cloud: {str(e)}")
            else:
                st.info("No neutral comments to analyze")
    
        with col3:
            st.markdown("#### Negative Comments")
            negative_comments = df_comments[df_comments['sentiment_category'] == 'Negative']
            if len(negative_comments) > 0:
                negative_tokens = negative_comments['tokens_list'].tolist()
                try:
                    fig_negative = create_wordcloud(negative_tokens, "Negative", wordcloud_negative_cmap)
                    st.pyplot(fig_negative)
                    plt.close(fig_negative)
                except Exception as e:
                    st.warning(f"Not enough negative words to generate word cloud: {str(e)}")
            else:
                st.info("No negative comments to analyze")
    
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    
        # Word Frequency Analysis
        st.markdown("### Top Words by Sentiment Category")
    
        # Calculate word frequencies
        from collections import Counter
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("#### Most Common in Positive")
            if len(positive_comments) > 0:
                all_positive_words = [word for tokens in positive_comments['tokens_list'] for word in tokens]
                positive_freq = Counter(all_positive_words).most_common(15)
    
                if positive_freq:
                    freq_df = pd.DataFrame(positive_freq, columns=['Word', 'Frequency'])
                    fig_pos_bar = px.bar(
                        freq_df,
                        x='Frequency',
                        y='Word',
                        orientation='h',
                        color_discrete_sequence=[color_positive or '#1E8449']
                    )
                    fig_pos_bar.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_pos_bar, width='stretch')
    
        with col2:
            st.markdown("#### Most Common in Neutral")
            if len(neutral_comments) > 0:
                all_neutral_words = [word for tokens in neutral_comments['tokens_list'] for word in tokens]
                neutral_freq = Counter(all_neutral_words).most_common(15)
    
                if neutral_freq:
                    freq_df = pd.DataFrame(neutral_freq, columns=['Word', 'Frequency'])
                    fig_neu_bar = px.bar(
                        freq_df,
                        x='Frequency',
                        y='Word',
                        orientation='h',
                        color_discrete_sequence=[color_neutral or '#707B7C']
                    )
                    fig_neu_bar.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_neu_bar, width='stretch')
    
        with col3:
            st.markdown("#### Most Common in Negative")
            if len(negative_comments) > 0:
                all_negative_words = [word for tokens in negative_comments['tokens_list'] for word in tokens]
                negative_freq = Counter(all_negative_words).most_common(15)
    
                if negative_freq:
                    freq_df = pd.DataFrame(negative_freq, columns=['Word', 'Frequency'])
                    fig_neg_bar = px.bar(
                        freq_df,
                        x='Frequency',
                        y='Word',
                        orientation='h',
                        color_discrete_sequence=[color_negative or '#C0392B']
                    )
                    fig_neg_bar.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_neg_bar, width='stretch')
    
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    
        # Topic Modeling
        st.markdown("### Topic Modeling Analysis")
        st.markdown("Discover the main themes and topics discussed in the comments using Latent Dirichlet Allocation (LDA).")
    
        with st.spinner("Building topic model... This may take a moment."):
            # Filter out empty token lists
            valid_documents = [tokens for tokens in df_comments['tokens_list'] if len(tokens) > 0]
    
            if len(valid_documents) >= 10:  # Need minimum documents for LDA
                try:
                    lda_results = build_lda_model(valid_documents, num_topics=num_topics)
                    topic_viz = get_topic_visualizations(lda_results, top_n_words=10)
    
                    # Display coherence score
                    st.metric(
                        "Model Coherence Score",
                        f"{topic_viz['coherence_score']:.3f}",
                        help="Coherence score measures how interpretable the topics are. Higher is better (typically 0.4-0.7 is good)."
                    )
    
                    # Display topics
                    st.markdown("#### Discovered Topics")
    
                    topic_cols = st.columns(min(3, num_topics))
                    for idx, topic in enumerate(topic_viz['topics']):
                        with topic_cols[idx % 3]:
                            st.markdown(f"**Topic {topic['id'] + 1}**")
                            st.write(", ".join(topic['words'][:8]))
    
                    # Interactive visualization
                    st.markdown("#### Interactive Topic Visualization")
                    st.markdown("Explore topics interactively. Click on topics to see their top terms and relationships.")
    
                    # Display pyLDAvis visualization
                    st.components.v1.html(topic_viz['html'], height=800, scrolling=True)
    
                except Exception as e:
                    st.error(f"Error building topic model: {str(e)}")
                    st.info("Try adjusting the number of topics in the sidebar or fetching more data.")
            else:
                st.warning(f"Not enough valid documents for topic modeling. Found {len(valid_documents)} documents, need at least 10. Try fetching more posts or reducing minimum comments filter.")
    
    # Tab 5: Data Export & Insights
    with tabs[4]:
        st.markdown('<p class="subheader">Data Export & Advanced Insights</p>', unsafe_allow_html=True)
    
        # Sentiment Trend Analysis
        st.markdown("### Sentiment Trend Analysis")
    
        if isinstance(sentiment_trend_info, dict):
            col1, col2, col3 = st.columns(3)
    
            with col1:
                trend_color = "#1E8449" if sentiment_trend_info['trend'] == "Up" else ("#C0392B" if sentiment_trend_info['trend'] == "Down" else "#707B7C")
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value" style="color:{trend_color}">{sentiment_trend_info["trend"]}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Sentiment Trend</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                change_color = "#1E8449" if sentiment_trend_info['latest_change'] > 0 else ("#C0392B" if sentiment_trend_info['latest_change'] < 0 else "#707B7C")
                st.markdown(f'<p class="metric-value" style="color:{change_color}">{sentiment_trend_info["latest_change"]:.1f}%</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Recent Change</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            with col3:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                value_color = "#1E8449" if sentiment_trend_info['latest_value'] > 0.05 else ("#C0392B" if sentiment_trend_info['latest_value'] < -0.05 else "#707B7C")
                st.markdown(f'<p class="metric-value" style="color:{value_color}">{sentiment_trend_info["latest_value"]:.3f}</p>', unsafe_allow_html=True)
                st.markdown('<p class="metric-label">Current Sentiment Score</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    
            st.info(sentiment_trend_info['description'])
        else:
            st.warning(sentiment_trend_info)
    
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    
        # Statistical Summary
        st.markdown("### Statistical Summary")
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown("#### Sentiment Score Statistics")
            # Calculate statistics directly to avoid pandas version issues
            vader_scores = df_comments['vader_compound']
            stats_df = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', '25%', '50% (Median)', '75%', 'Max'],
                'Value': [
                    f"{vader_scores.count():.0f}",
                    f"{vader_scores.mean():.4f}",
                    f"{vader_scores.std():.4f}",
                    f"{vader_scores.min():.4f}",
                    f"{vader_scores.quantile(0.25):.4f}",
                    f"{vader_scores.median():.4f}",
                    f"{vader_scores.quantile(0.75):.4f}",
                    f"{vader_scores.max():.4f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, width='stretch')
    
        with col2:
            st.markdown("#### Comment Activity Statistics")
    
            # Calculate additional metrics
            avg_score = df_comments['comment_score'].mean()
            total_posts = len(df_posts)
            avg_comments_per_post = len(df_comments) / total_posts if total_posts > 0 else 0
    
            # Date range
            date_range = df_comments['comment_date'].max() - df_comments['comment_date'].min()
    
            activity_df = pd.DataFrame({
                'Metric': [
                    'Total Posts Analyzed',
                    'Avg Comments per Post',
                    'Avg Comment Score',
                    'Date Range (days)',
                    'Most Active Day'
                ],
                'Value': [
                    f"{total_posts}",
                    f"{avg_comments_per_post:.1f}",
                    f"{avg_score:.1f}",
                    f"{date_range.days}",
                    f"{df_comments.groupby('comment_date').size().idxmax()}"
                ]
            })
            st.dataframe(activity_df, hide_index=True, width='stretch')
    
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    
        # Correlation Analysis
        st.markdown("### Correlation Analysis")
        st.markdown("Explore relationships between different metrics")
    
        col1, col2 = st.columns(2)
    
        with col1:
            # Sentiment vs Comment Score
            fig_corr1 = px.scatter(
                df_comments,
                x='comment_score',
                y='vader_compound',
                color='sentiment_category',
                color_discrete_map={
                    'Positive': color_positive or '#1E8449',
                    'Neutral': color_neutral or '#707B7C',
                    'Negative': color_negative or '#C0392B'
                },
                title="Sentiment vs Comment Score",
                trendline="ols",  # Changed from lowess to ols (works without statsmodels)
                opacity=0.6
            )
            fig_corr1.update_layout(
                xaxis_title="Comment Score (Upvotes)",
                yaxis_title="Sentiment Score",
                height=400
            )
            st.plotly_chart(fig_corr1, width='stretch')
    
            # Calculate correlation
            corr_score_sentiment = df_comments['comment_score'].corr(df_comments['vader_compound'])
            st.metric("Correlation Coefficient", f"{corr_score_sentiment:.3f}",
                     help="Pearson correlation between comment score and sentiment (-1 to 1)")
    
        with col2:
            # Average sentiment by post engagement
            post_engagement = df_comments.groupby('post_id').agg({
                'vader_compound': 'mean',
                'comment_id': 'count'
            }).reset_index()
            post_engagement.columns = ['Post ID', 'Avg Sentiment', 'Comment Count']
    
            fig_corr2 = px.scatter(
                post_engagement,
                x='Comment Count',
                y='Avg Sentiment',
                title="Post Engagement vs Average Sentiment",
                trendline="ols",  # Changed from lowess to ols (works without statsmodels)
                color='Avg Sentiment',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                opacity=0.7
            )
            fig_corr2.update_layout(
                xaxis_title="Number of Comments",
                yaxis_title="Average Sentiment Score",
                height=400
            )
            st.plotly_chart(fig_corr2, width='stretch')
    
            corr_engagement_sentiment = post_engagement['Comment Count'].corr(post_engagement['Avg Sentiment'])
            st.metric("Correlation Coefficient", f"{corr_engagement_sentiment:.3f}",
                     help="Pearson correlation between post engagement and sentiment (-1 to 1)")
    
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    
        # Data Export Section
        st.markdown("### Export Data")
        st.markdown("Download the analyzed data in various formats for further analysis.")
    
        col1, col2, col3 = st.columns(3)
    
        with col1:
            st.markdown("#### Comments Data")
            # Prepare clean export dataframe
            export_df_comments = df_comments[[
                'comment_id', 'comment_text', 'comment_author', 'comment_score',
                'comment_datetime', 'vader_compound', 'sentiment_category',
                'post_id', 'post_title'
            ]].copy()
    
            csv_comments = export_df_comments.to_csv(index=False)
            st.download_button(
                label="Download Comments CSV",
                data=csv_comments,
                file_name=f"tesla_sentiment_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all analyzed comments with sentiment scores"
            )
    
        with col2:
            st.markdown("#### Posts Data")
            export_df_posts = df_posts[[
                'post_id', 'post_title', 'post_url', 'post_score',
                'post_upvote_ratio', 'post_num_comments', 'post_author'
            ]].copy()
    
            csv_posts = export_df_posts.to_csv(index=False)
            st.download_button(
                label="Download Posts CSV",
                data=csv_posts,
                file_name=f"tesla_sentiment_posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all posts metadata"
            )
    
        with col3:
            st.markdown("#### Summary Report")
    
            # Create comprehensive summary
            summary_report = {
                "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subreddit": subreddit_name,
                "data_collection": {
                    "total_posts": int(len(df_posts)),
                    "total_comments": int(len(df_comments)),
                    "date_range_start": str(df_comments['comment_date'].min()),
                    "date_range_end": str(df_comments['comment_date'].max())
                },
                "sentiment_summary": {
                    "average_sentiment": float(sentiment_summary['avg_compound']),
                    "positive_percentage": float(sentiment_summary['positive_pct']),
                    "neutral_percentage": float(sentiment_summary['neutral_pct']),
                    "negative_percentage": float(sentiment_summary['negative_pct'])
                },
                "sentiment_trend": sentiment_trend_info if isinstance(sentiment_trend_info, dict) else str(sentiment_trend_info),
                "statistics": {
                    "mean": float(df_comments['vader_compound'].mean()),
                    "median": float(df_comments['vader_compound'].median()),
                    "std_dev": float(df_comments['vader_compound'].std()),
                    "min": float(df_comments['vader_compound'].min()),
                    "max": float(df_comments['vader_compound'].max())
                },
                "correlations": {
                    "sentiment_vs_score": float(corr_score_sentiment),
                    "engagement_vs_sentiment": float(corr_engagement_sentiment)
                }
            }
    
            json_report = json.dumps(summary_report, indent=2)
            st.download_button(
                label="Download JSON Report",
                data=json_report,
                file_name=f"tesla_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Download comprehensive analysis report in JSON format"
            )
    
        st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    
        # Raw Data Preview
        st.markdown("### Data Preview")
    
        preview_option = st.selectbox(
            "Select data to preview:",
            ["Comments with Sentiment", "Posts Summary", "Sentiment by Date"]
        )
    
        if preview_option == "Comments with Sentiment":
            st.dataframe(
                export_df_comments.head(100),
                width='stretch',
                hide_index=True
            )
        elif preview_option == "Posts Summary":
            st.dataframe(
                post_sentiment[[
                    'Post Title', 'Avg Sentiment', 'sentiment_category',
                    'Comment Count', 'post_score', 'post_upvote_ratio'
                ]].head(50),
                width='stretch',
                hide_index=True
            )
        else:  # Sentiment by Date
            sentiment_by_date_detailed = df_comments.groupby('comment_date').agg({
                'vader_compound': ['mean', 'std', 'count'],
                'sentiment_category': lambda x: (x == 'Positive').sum() / len(x) * 100
            }).reset_index()
            sentiment_by_date_detailed.columns = ['Date', 'Avg Sentiment', 'Std Dev', 'Comment Count', 'Positive %']
            st.dataframe(
                sentiment_by_date_detailed,
                width='stretch',
                hide_index=True
            )
    
    # --- Professional Footer ---
    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <h3 style="margin-top: 0; color: #ecf0f1;">⚡ Tesla Sentiment Dashboard</h3>
        <p style="font-size: 1rem; margin: 10px 0;">
            Análisis de sentiment en tiempo real usando VADER (NLTK) y visualización avanzada con Plotly
        </p>
        <div class="footer-links">
            <a href="https://github.com/jburguera/reddit-sentiment-dashboard-streamlit" target="_blank">📁 GitHub</a> •
            <a href="https://www.linkedin.com/in/jburguera/" target="_blank">💼 LinkedIn</a> •
            <a href="https://www.reddit.com/r/{}" target="_blank">🔗 r/{}</a>
        </div>
        <div class="footer-info">
            <p style="margin: 5px 0;">
                <strong>Última actualización:</strong> {} |
                <strong>Versión:</strong> 1.0.0 |
                <strong>Posts analizados:</strong> {}
            </p>
            <p style="margin: 5px 0; font-size: 0.85rem;">
                Hecho con ❤️ usando Streamlit, PRAW, NLTK, y Plotly
            </p>
            <p style="margin: 5px 0; font-size: 0.8rem; color: #7f8c8d;">
                © 2025 | Los datos provienen de Reddit y se analizan en tiempo real
            </p>
        </div>
    </div>
    """.format(
        subreddit_name,
        subreddit_name,
        st.session_state.get('last_refresh', datetime.now()).strftime("%Y-%m-%d %H:%M:%S UTC"),
        len(df_posts)
    ), unsafe_allow_html=True)

else:
    # Show welcome message when no analysis has been run yet
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px;">
        <h2 style="color: #E31937;">Bienvenido al Dashboard de Sentiment Tesla</h2>
        <p style="font-size: 1.2rem; margin: 30px 0;">
            ¿Listo para analizar el sentiment en Reddit?
        </p>
        <p style="color: #666; margin: 20px 0;">
            Configura los parámetros en la barra lateral y haz click en
            <strong>"🔍 Analyze Sentiment"</strong> para empezar.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Show sample information cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <h3>📊 Análisis de sentiment</h3>
            <p>Analiza miles de comentarios de Reddit para entender la opinión pública sobre Tesla usando VADER.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>📈 Detección de tendencias</h3>
            <p>Rastrea cómo evoluciona el sentiment en el tiempo e identifica cambios con análisis visual.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <h3>💬 Modelado de tópicos</h3>
            <p>Descubre los temas principales de conversación usando modelado LDA.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="elegant-separator">', unsafe_allow_html=True)

    # Instructions
    st.markdown("### Cómo empezar")
    st.markdown("""
    1. **Configurar parámetros** (en la barra lateral):
       - Elige el subreddit a analizar
       - Selecciona cuántos posts descargar
       - Define el período de tiempo
       - Personaliza colores y temas

    2. **Iniciar análisis**:
       - Click en **"🔍 Analyze Sentiment"**
       - Espera a que se descarguen y analicen los datos

    3. **Explorar resultados**:
       - Ver métricas y tendencias de sentiment
       - Analizar los posts más destacados
       - Explorar nubes de palabras y modelos de tópicos
       - Exportar datos para análisis adicional
    """)

