# Tesla Sentiment Dashboard

What does Reddit say about Tesla? This dashboard analyzes thousands of comments in real time to measure community sentiment toward the brand, identify trends, and discover what people are talking about when they mention Tesla.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-green?style=for-the-badge)

## Why this project exists

Social media is a thermometer of public sentiment. Reddit, with its active communities and candid debates, offers a privileged window into how people perceive Tesla: from enthusiasts celebrating every innovation to critics questioning each decision.

This dashboard automates that analysis. Instead of manually reading hundreds of comments, it uses natural language processing techniques to classify them, detect patterns, and present results in visualizations that tell the story.

## What it actually does

### Measures sentiment (positive, negative, or neutral)
Each comment goes through VADER, an algorithm specialized in analyzing social media text. VADER assigns a score between -1 (very negative) and +1 (very positive). The dashboard automatically classifies comments and displays:
- Pie charts showing the percentage of each category
- Histograms showing score distribution
- Visual comparisons to understand the overall balance

### Detects temporal trends
Is sentiment improving or worsening? The dashboard tracks evolution over time:
- Line graphs showing day-to-day changes
- Analysis by hour to identify when conversation is most active
- Trend indicators that signal whether sentiment is rising, falling, or holding steady

### Identifies conversation topics
Using topic modeling (a technique that groups related words), the dashboard automatically discovers what people are talking about:
- Are they discussing autopilot?
- Talking about prices and stock?
- Focusing on build quality?

### Highlights what matters most
The dashboard sorts and filters to show:
- Posts that generated the most debate
- The most positive and most negative comments
- Words most repeated in each sentiment category

### Exports everything for additional analysis
All data can be downloaded in standard formats (CSV and JSON) for anyone who wants to dig deeper or create their own analyses.

## How to install it

### What you need

- Python 3.8 or newer
- A Reddit account (free) to access their API
- 10 minutes to set everything up

### Installation steps

**1. Download the code**
```bash
git clone https://github.com/jburguera/reddit-sentiment-dashboard-streamlit.git
cd reddit-sentiment-dashboard-streamlit
```

**2. Install required libraries**
```bash
pip install -r requirements.txt
```

**3. Get your Reddit credentials**

Reddit requires you to identify yourself to use their API (this is how they control usage and prevent abuse):

- Go to https://www.reddit.com/prefs/apps
- Click "Create App" or "Create Another App"
- Fill out the form:
  - **name**: whatever you want (e.g., "sentiment-analyzer")
  - **type**: select "script"
  - **redirect uri**: enter http://localhost:8080 (required but not used)
- Once created, you'll see two important codes:
  - The **client_id** (appears just below the name)
  - The **client_secret** (appears labeled as "secret")

**4. Create your credentials file**

In the project folder, create a file named `reddit_credentials.txt` with this content:

```
CLIENT_ID=your_client_id_here
CLIENT_SECRET=your_client_secret_here
USER_AGENT=tesla-sentiment-dashboard/1.0
```

Replace the values with the codes you got from Reddit.

**5. Start the dashboard**
```bash
streamlit run streamlit_app.py
```

The browser will open automatically at `http://localhost:8501`

## How to use it

**Basic configuration**

The left sidebar has all the controls:

1. **Subreddit**: By default it analyzes r/TeslaMotors, but you can enter any other (without the "r/")
2. **Number of posts**: More posts means more data but also longer wait times (recommended: 200)
3. **Minimum comments**: Filters posts with little activity (recommended: 150)
4. **Time period**: Analyzes recent posts (day, week, month...)

**Analysis**

Click "🔍 Analyze Sentiment" and wait. The dashboard will keep you informed of progress:
- Downloading posts from Reddit
- Analyzing sentiment with VADER
- Preprocessing text for advanced analysis

Once finished, you'll have access to five main sections with all the information.

## How it works under the hood

### Text processing

Before analyzing, the dashboard cleans and prepares each comment:

1. **Normalization**: Everything to lowercase so "Tesla" and "tesla" are treated the same
2. **Cleaning**: Removes URLs, mentions, and special characters that don't add meaning
3. **Tokenization**: Divides text into individual words
4. **Filtering**: Removes stop words (the, a, of, that...) and very common Reddit terms (edit, deleted...)
5. **Lemmatization**: Reduces words to their root (buying → buy, cars → car)

This process improves analysis quality by focusing on the actual content of messages.

### Sentiment analysis

VADER (Valence Aware Dictionary and Sentiment Reasoner) is especially good with informal social media text because:

- It understands capitalization as emphasis ("AMAZING" carries more weight than "amazing")
- Recognizes emojis and emoticons
- Detects negations ("not bad" vs "bad")
- Handles intensifiers ("very good" vs "good")

The result is a score that the dashboard classifies like this:
- **Positive**: ≥ 0.05
- **Neutral**: between -0.05 and 0.05
- **Negative**: ≤ -0.05

### Topic modeling

This is perhaps the most sophisticated part. The dashboard uses LDA (Latent Dirichlet Allocation), an algorithm that:

1. Groups words that tend to appear together
2. Identifies thematic patterns without human supervision
3. Assigns each comment to one or several topics

For example, it might automatically discover there's a group of comments about "price, stock, investment, market" (financial topic) and another about "battery, range, charging, miles" (technical topic).

The "coherence score" measures how interpretable the discovered topics are. Values between 0.4 and 0.7 are good.

### Trend detection

The dashboard calculates 3-day moving averages to smooth out fluctuations:

- **Upward trend**: sentiment increased more than 5%
- **Downward trend**: sentiment fell more than 5%
- **Stable**: changes less than 5%

## Project structure

```
reddit-sentiment-dashboard-streamlit/
├── streamlit_app.py           # All dashboard logic
├── requirements.txt           # Required libraries (Streamlit, NLTK, etc.)
├── reddit_credentials.txt     # Your credentials (DO NOT upload to git)
├── README.md                  # This guide
└── nltk_data/                 # Language data (downloads automatically)
```

## Libraries it uses

- **Streamlit**: Converts Python scripts into interactive web applications
- **PRAW**: Python client for Reddit's API
- **NLTK**: Specialized library for natural language processing
- **VADER**: Sentiment analysis algorithm included in NLTK
- **Gensim**: Efficient implementation of LDA for topic modeling
- **Plotly**: Interactive charts (zoom, hover, export)
- **Pandas**: Manipulation and analysis of tabular data
- **WordCloud**: Generation of visual word clouds

## Troubleshooting common issues

### "NLTK resources not found"

The app tries to download what's needed automatically, but if it fails:
```bash
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords')"
```

### "Error: 429 - Rate limit exceeded"

Reddit's API has usage limits. If you see this error:
- Reduce the number of posts to analyze
- Wait 5-10 minutes before trying again
- Verify your credentials are correct

### "Not enough documents for topic modeling"

Topic modeling needs at least 10 comments with content. Solution:
- Analyze more posts
- Reduce minimum comments per post
- Choose a more active subreddit

### The app is very slow or runs out of memory

Analyzing thousands of comments consumes resources:
- Reduce post limit to 100-150
- Close other heavy applications
- In the Streamlit menu (top right), select "Clear cache"

## Possible future improvements

This is a living project. Some ideas to expand it:

- Add aspect-based sentiment analysis (price, quality, service...)
- Compare sentiment across different subreddits
- Detect bots or suspicious accounts
- Sentiment analysis by user (who's most positive/negative)
- Automatic alerts when there are sudden changes
- Export automatic PDF reports

## Credits and acknowledgments

This project wouldn't be possible without:

- **NLTK**: The library that democratized natural language processing
- **PRAW**: For making Reddit's API accessible
- **Streamlit**: For simplifying the creation of interactive dashboards
- **The Reddit community**: For generating rich conversations to analyze

---

**License**: MIT (use it, modify it, share it freely)

**Built with**: Python 3.11, Streamlit, lots of coffee, and curiosity about what the internet thinks of Tesla

**Author**: [Javier Burguera](https://www.linkedin.com/in/jburguera/) | [GitHub](https://github.com/jburguera/reddit-sentiment-dashboard-streamlit)
