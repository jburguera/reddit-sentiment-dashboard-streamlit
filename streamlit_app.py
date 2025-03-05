import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import praw
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
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
    .main-header {
        font-size: 2.5rem;
        color: #E31937;
        font-weight: 800;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1rem;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #666666;
        text-align: center;
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
        color: #1E8449;
        font-weight: 600;
    }
    .sentiment-neutral {
        color: #707B7C;
        font-weight: 600;
    }
    .sentiment-negative {
        color: #C0392B;
        font-weight: 600;
    }
    .trend-up {
        color: #1E8449;
        font-weight: 600;
    }
    .trend-stable {
        color: #707B7C;
        font-weight: 600;
    }
    .trend-down {
        color: #C0392B;
        font-weight: 600;
    }
    hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .download-btn {
        background-color: #E31937;
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: 600;
        display: inline-block;
        text-align: center;
        margin-top: 10px;
    }
    .download-btn:hover {
        background-color: #C0392B;
        color: white;
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
@st.cache_data(ttl=3600, show_spinner=False)
def download_nltk_resources():
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')  
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Explicitly set NLTK data path
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)
    
    resources = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
            # Verify download
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt/english.pickle')
        except LookupError:
                nltk.data.find('tokenizers/punkt_tab/english/')
        except Exception as e:
            st.error(f"Error downloading {resource}: {str(e)}")
            return False
    return True

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
    
    # Tokenize
    tokens = word_tokenize(text)
    
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
        
        # Create progress bar
        progress_text = "Fetching Reddit data..."
        progress_bar = st.progress(0, text=progress_text)
        
        # Get hot posts
        hot_posts = list(subreddit.hot(limit=post_limit))
        filtered_posts = [post for post in hot_posts if post.num_comments >= min_comments]
        
        # Check if we have posts
        if not filtered_posts:
            st.warning(f"No posts found with at least {min_comments} comments. Try adjusting your filters.")
            return None, None
        
        # Prepare for data collection
        post_data_list = []
        comment_data_list = []
        
        # Process posts with progress tracking
        total_posts = len(filtered_posts)
        for i, post in enumerate(filtered_posts):
            # Update progress
            progress = int((i + 1) / total_posts * 100)
            progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_posts} posts)")
            
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
        
        # Clear progress bar
        progress_bar.empty()
        
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
    
    # Create progress bar
    progress_text = "Analyzing sentiment..."
    progress_bar = st.progress(0, text=progress_text)
    
    # Prepare columns for sentiment scores
    df_comments['vader_neg'] = None
    df_comments['vader_neu'] = None
    df_comments['vader_pos'] = None
    df_comments['vader_compound'] = None
    
    # Process comments with progress tracking
    total_comments = len(df_comments)
    for i, (index, row) in enumerate(df_comments.iterrows()):
        # Update progress every 10 comments to improve performance
        if i % 10 == 0 or i == total_comments - 1:
            progress = int((i + 1) / total_comments * 100)
            progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_comments} comments)")
        
        comment_text = row['comment_text']
        if isinstance(comment_text, str):
            vs = analyzer.polarity_scores(comment_text)
            df_comments.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = vs.values()
        else:
            df_comments.loc[index, ['vader_neg', 'vader_neu', 'vader_pos', 'vader_compound']] = [0.0, 1.0, 0.0, 0.0]
    
    # Clear progress bar
    progress_bar.empty()
    
    # Categorize sentiment
    df_comments['sentiment_category'] = df_comments['vader_compound'].apply(
        lambda score: "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
    )
    
    # Add datetime columns
    df_comments['comment_datetime'] = pd.to_datetime(df_comments['comment_created_utc'], unit='s', utc=True)
    df_comments['comment_date'] = df_comments['comment_datetime'].dt.date
    df_comments['comment_hour'] = df_comments['comment_datetime'].dt.hour
    
    # Preprocess comment text for NLP analysis
    progress_text = "Preprocessing text for NLP analysis..."
    progress_bar = st.progress(0, text=progress_text)
    
    df_comments['tokens'] = None
    total_comments = len(df_comments)
    
    for i, (index, row) in enumerate(df_comments.iterrows()):
        if i % 20 == 0 or i == total_comments - 1:
            progress = int((i + 1) / total_comments * 100)
            progress_bar.progress(progress, text=f"{progress_text} ({i+1}/{total_comments} comments)")
        
        tokens = preprocess_text(row['comment_text'])
        df_comments.loc[index, 'tokens'] = str(tokens)  # Store as string for serialization
    
    progress_bar.empty()
    
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
    st.markdown("## Dashboard Settings")
    st.markdown("---")
    
    # Reddit Configuration
    st.markdown("### Reddit Data Source")
    subreddit_name = st.text_input("Subreddit", "TeslaMotors", help="Enter the subreddit name without r/")
    
    post_limit = st.slider(
        "Number of posts to fetch",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="More posts means more data but slower loading time"
    )
    
    min_comments = st.slider(
        "Minimum comments per post",
        min_value=10,
        max_value=300,
        value=150,
        step=10,
        help="Filter posts by minimum number of comments"
    )
    
    time_filter = st.selectbox(
        "Time period",
        ["day", "week", "month", "year", "all"],
        index=1,
        help="Time period for hot posts"
    )
    
    st.markdown("---")
    
    # Visualization Settings
    st.markdown("### Visualization Settings")
    
    color_theme = st.selectbox(
        "Color Theme", 
        ["Tesla", "Viridis", "Plasma", "Inferno", "Magma"],
        index=0
    )
    
    chart_style = st.selectbox(
        "Chart Style",
        ["whitegrid", "darkgrid", "white", "dark", "ticks"],
        index=0
    )
    
    num_topics = st.slider(
        "Number of topics for topic modeling",
        min_value=2,
        max_value=10,
        value=5,
        help="More topics will show finer distinctions but may be harder to interpret"
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
    
    st.markdown("---")
    
    # Auto-refresh settings
    st.markdown("### Auto-Refresh Settings")
    
    # Initialize auto-refresh state if not exists
    if 'auto_refresh_enabled' not in st.session_state:
        st.session_state['auto_refresh_enabled'] = False
    
    auto_refresh = st.checkbox(
        "Enable auto-refresh", 
        value=st.session_state.get('auto_refresh_enabled', False),
        help="Automatically refresh data at specified intervals"
    )
    
    refresh_interval_options = {
        "30 minutes": 1800,
        "1 hour": 3600,
        "3 hours": 10800,
        "6 hours": 21600,
        "12 hours": 43200,
        "24 hours": 86400
    }
    
    refresh_interval_key = st.selectbox(
        "Refresh interval",
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
    
    st.markdown("---")
    
    # Manual refresh button
    refresh_data = st.button("Refresh Data Now", key="refresh_data")
    
    st.markdown("---")
    st.markdown("#### About")
    st.markdown("This dashboard analyzes sentiment in Reddit posts about Tesla from the selected subreddit.")
    st.markdown("Data is automatically refreshed at your chosen interval.")

# --- Main Content ---
st.markdown('<p class="main-header">Tesla Sentiment Dashboard</p>', unsafe_allow_html=True)

st.markdown("""
This interactive dashboard analyzes public sentiment towards Tesla on Reddit. 
The analysis uses VADER sentiment analysis from NLTK to categorize comments as positive, negative, or neutral.
""")

# --- Main Process ---
# Download NLTK resources
if not download_nltk_resources():
    st.error("Failed to download required NLTK resources. Please check your internet connection.")
    st.stop()

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

# Check if we should refresh data
if 'last_refresh' not in st.session_state or refresh_data:
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

# --- Create summary metrics ---
sentiment_summary = create_sentiment_summary(df_comments)

# Display summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-value">{sentiment_summary["total_comments"]}</p>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Total Comments</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    sentiment_score = sentiment_summary["avg_compound"]
    sentiment_color = "#1E8449" if sentiment_score >= 0.05 else ("#C0392B" if sentiment_score <= -0.05 else "#707B7C")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-value" style="color:{sentiment_color}">{sentiment_score:.2f}</p>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Average Sentiment Score</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-value sentiment-positive">{sentiment_summary["positive_pct"]:.1f}%</p>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Positive Comments</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<p class="metric-value sentiment-negative">{sentiment_summary["negative_pct"]:.1f}%</p>', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Negative Comments</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Visualization Tabs ---
tabs = st.tabs(["Sentiment Distribution", "Temporal Analysis", "Top Posts", "Word Analysis"])

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
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
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
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
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
    
    st.plotly_chart(fig_violin, use_container_width=True)

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
    
    st.plotly_chart(fig_line, use_container_width=True)
    
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
    
    st.plotly_chart(fig_hour, use_container_width=True)
    
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
    
    st.plotly_chart(fig_volume, use_container_width=True)

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
    st.markdown("""
    This tab would typically contain word clouds, frequency analysis, and topic modeling.
    These features require additional NLP libraries like NLTK's tokenizers and stopwords.
    
    For a complete implementation, consider adding:
    - Word clouds by sentiment category
    - Frequency analysis of most common terms
    - Topic modeling to identify main discussion themes
    - Named entity recognition for key products/people mentioned
    """)
    
    st.info("To implement these features, you would need to add code to download additional NLTK resources and use libraries like WordCloud and Gensim.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Tesla Sentiment Dashboard | Updated: {}</p>
    <p>Data source: Reddit r/{}</p>
</div>
""".format(
    st.session_state.get('last_refresh', datetime.now()).strftime("%Y-%m-%d %H:%M:%S UTC"),
    subreddit_name
), unsafe_allow_html=True)

        
