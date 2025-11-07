# Tesla Sentiment Dashboard - Reddit Analytics

A comprehensive, interactive dashboard for analyzing sentiment towards Tesla on Reddit using advanced NLP techniques and real-time data visualization.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Processing-green?style=for-the-badge)

## Features

### 🎯 Core Functionality

- **Real-time Reddit Data Collection**: Fetches posts and comments from any subreddit (default: r/TeslaMotors)
- **VADER Sentiment Analysis**: Uses NLTK's VADER for accurate sentiment scoring
- **Interactive Visualizations**: Beautiful, interactive charts using Plotly and Matplotlib
- **Topic Modeling**: LDA-based topic discovery to identify discussion themes
- **Word Cloud Analysis**: Visual representation of most common words by sentiment
- **Trend Detection**: Automatic detection of sentiment trends over time
- **Data Export**: Export analyzed data in CSV and JSON formats

### 📊 Dashboard Sections

#### 1. Sentiment Distribution
- Pie chart showing distribution of positive/neutral/negative comments
- Histogram of sentiment scores
- Violin plots for detailed score distribution
- Summary metrics cards

#### 2. Temporal Analysis
- Sentiment trends over time with threshold indicators
- Hourly sentiment patterns
- Comment volume tracking
- Interactive time-series visualizations

#### 3. Top Posts Analysis
- Most positive posts with links
- Most negative posts with detailed metrics
- Most discussed posts
- Engagement metrics (upvotes, comments, ratios)

#### 4. Word Analysis
- Word clouds by sentiment category (Positive, Neutral, Negative)
- Top 15 most common words per sentiment
- Interactive topic modeling with pyLDAvis
- Coherence score for topic quality
- Dynamic topic discovery (2-10 topics)

#### 5. Data Export & Advanced Insights
- **Sentiment Trend Indicators**: Real-time trend direction with percentage changes
- **Statistical Summary**: Comprehensive statistics (mean, median, std dev, quartiles)
- **Correlation Analysis**:
  - Sentiment vs Comment Score
  - Post Engagement vs Sentiment
  - Interactive scatter plots with trendlines
- **Data Export Options**:
  - Comments CSV with full sentiment data
  - Posts CSV with metadata
  - JSON report with comprehensive analytics
- **Data Preview**: Browse comments, posts, and aggregated data

### 🎨 Customization Options

- **Subreddit Selection**: Analyze any subreddit
- **Post Filtering**: Adjust number of posts (50-500) and minimum comments
- **Time Periods**: Day, week, month, year, or all-time
- **Color Themes**: Tesla, Viridis, Plasma, Inferno, Magma
- **Chart Styles**: Multiple seaborn style options
- **Topic Count**: Configurable topic modeling (2-10 topics)
- **Auto-Refresh**: Automatic data updates (30 min to 24 hours)

### ✨ Enhanced Features

- **Splash Screen**: Beautiful loading screen on first launch
- **Progress Tracking**: Real-time progress bars for data fetching and analysis
- **Error Handling**: Robust error handling with helpful messages
- **Caching**: Smart caching for improved performance (15-minute TTL)
- **Responsive Design**: Mobile-friendly layout
- **Custom CSS**: Tesla-branded design with custom colors

## Installation

### Prerequisites

- Python 3.8 or higher
- Reddit API credentials (free)

### Setup Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/reddit-sentiment-dashboard-streamlit.git
cd reddit-sentiment-dashboard-streamlit
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Create Reddit API credentials**:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Select "script" as the app type
   - Fill in the required fields
   - Note your `client_id` and `client_secret`

4. **Create credentials file**:
Create a file named `reddit_credentials.txt` in the project directory with the following format:
```
CLIENT_ID=your_client_id_here
CLIENT_SECRET=your_client_secret_here
USER_AGENT=your_app_name/1.0
```

5. **Run the dashboard**:
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your default browser at `http://localhost:8501`

## Usage

### Basic Usage

1. **Configure Settings** (Sidebar):
   - Enter the subreddit name (without r/)
   - Adjust the number of posts to fetch
   - Set minimum comments filter
   - Select time period

2. **Customize Visualization**:
   - Choose color theme
   - Select chart style
   - Set number of topics for modeling

3. **Enable Auto-Refresh** (Optional):
   - Toggle auto-refresh
   - Select refresh interval
   - Monitor next refresh time

4. **Explore Tabs**:
   - Navigate through different analysis sections
   - Interact with visualizations
   - Export data as needed

### Advanced Usage

#### Topic Modeling
The dashboard uses Latent Dirichlet Allocation (LDA) to discover topics:
- Adjust the number of topics in the sidebar (2-10)
- Higher coherence scores (0.4-0.7) indicate better topics
- Click on topics in the interactive visualization to explore

#### Trend Analysis
The sentiment trend detector analyzes rolling averages:
- "Up" trend: >5% increase in sentiment
- "Down" trend: >5% decrease in sentiment
- "Stable": Changes within ±5%

#### Data Export
Export options include:
- **Comments CSV**: All comments with sentiment scores
- **Posts CSV**: Post metadata and metrics
- **JSON Report**: Comprehensive analysis summary

## Dependencies

```
streamlit>=1.22.0       # Web framework
matplotlib              # Static plotting
seaborn                 # Statistical visualization
pandas                  # Data manipulation
praw                    # Reddit API wrapper
nltk>=3.7              # NLP toolkit
wordcloud              # Word cloud generation
gensim                 # Topic modeling
pyLDAvis               # LDA visualization
plotly                 # Interactive plots
Pillow                 # Image processing
```

## Project Structure

```
reddit-sentiment-dashboard-streamlit/
├── streamlit_app.py           # Main application file
├── requirements.txt           # Python dependencies
├── reddit_credentials.txt     # API credentials (create this)
├── README.md                  # This file
├── LICENSE                    # License file
└── nltk_data/                # Auto-created NLTK data directory
```

## Features in Detail

### NLP Pipeline

1. **Text Preprocessing**:
   - Lowercase conversion
   - URL and special character removal
   - Tokenization
   - Stopword removal (custom + Reddit-specific)
   - Lemmatization

2. **Sentiment Analysis**:
   - VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Compound score: -1 (most negative) to +1 (most positive)
   - Classification: Positive (≥0.05), Neutral (-0.05 to 0.05), Negative (≤-0.05)

3. **Topic Modeling**:
   - LDA with automatic parameter tuning
   - Coherence score calculation (C_v metric)
   - Interactive pyLDAvis visualization
   - Extreme value filtering for better topics

### Performance Optimizations

- **Caching**: Functions cached with appropriate TTL
- **Progress Tracking**: Updates every 10 comments for smooth UX
- **Lazy Loading**: Data fetched only when needed
- **Efficient Processing**: Vectorized pandas operations

## Troubleshooting

### Common Issues

1. **NLTK Resources Not Found**:
   - The app automatically downloads required NLTK data
   - If issues persist, manually run: `python -c "import nltk; nltk.download('all')"`

2. **Reddit API Rate Limiting**:
   - Reduce number of posts to fetch
   - Increase minimum comments filter
   - Wait before making new requests

3. **Topic Modeling Errors**:
   - Ensure you have at least 10 valid documents
   - Try fetching more posts
   - Reduce minimum comments filter
   - Adjust number of topics

4. **Memory Issues**:
   - Reduce post limit
   - Clear cache: Settings → Clear cache
   - Restart the Streamlit app

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

1. Follow PEP 8 style guide
2. Add docstrings to functions
3. Test changes thoroughly
4. Update README if adding features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NLTK**: Natural Language Toolkit for NLP functionality
- **PRAW**: Python Reddit API Wrapper
- **Streamlit**: Amazing framework for data apps
- **Plotly**: Interactive visualization library
- **Gensim**: Topic modeling library

## Future Enhancements

Potential features for future versions:
- [ ] Multi-subreddit comparison
- [ ] Historical data tracking and storage
- [ ] Named Entity Recognition (NER)
- [ ] Aspect-based sentiment analysis
- [ ] Email alerts for sentiment changes
- [ ] API endpoint for external integrations
- [ ] Machine learning model fine-tuning
- [ ] Competitor comparison (Tesla vs other EVs)
- [ ] Social network analysis
- [ ] Export to PowerPoint/PDF reports

## Screenshots

(Add screenshots of your dashboard here)

## Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

**Made with ❤️ using Streamlit and Python**
