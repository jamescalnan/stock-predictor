import csv
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.logging import RichHandler
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

FORMAT = "%(asctime)s — %(message)s"
logging.basicConfig(
    level="INFO", 
    format=FORMAT, 
    datefmt="[%X]", 
    handlers=[RichHandler()]
)

logger = logging.getLogger("rich")
import requests

c = Console()

def get_news(ticker):
    API_ENDPOINT = "https://newsapi.org/v2/everything"
    API_KEY = "ef2e0d24af4f471bbd9d1b34c01d1360"  
    query = ticker 
    
    params = {
        'q': query,
        'apiKey': API_KEY,
    }
    
    logger.info('Fetching news...')

    response = requests.get(API_ENDPOINT, params=params)
    news_data = response.json()

    # Filter out articles from "www.fool.com"
    filtered_articles = [article for article in news_data["articles"] if "www.fool.com" not in article["url"]]

    logger.info(f'News fetched, found {len(filtered_articles)} articles (excluding www.fool.com).')

    return filtered_articles




def get_sentiment_score(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    
    # VADER's 'compound' score is computed by summing the valence scores of each word in the lexicon, 
    # adjusted according to the rules, and then normalized between -1 (most extreme negative) and +1 (most extreme positive).
    return sentiment['compound']




def save_to_csv(data, ticker):
    file_path = f'news_sentiment_data/{ticker.replace("$", "")}_news_data.csv'
    
    # Step 1: Read existing data
    logger.info('Reading existing data...')
    existing_data = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            existing_data = [row for row in reader]
    except FileNotFoundError:
        # If file doesn't exist, it will be created later.
        logger.info(f'No existing data found at {file_path}.')
        pass

    # Collect URLs from existing data for comparison.
    if len(existing_data) > 0:
        logger.info('Collecting existing URLs...')
    existing_urls = [row[3] for row in existing_data]
    
    new_articles = 0

    with open(file_path, 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        for row in data:
            # Step 2: Check if the URL is already in the existing data
            if row[3] not in existing_urls:  # Assuming URL is at index 3
                writer.writerow(row)
                new_articles += 1

    logger.info(f'{new_articles} new articles saved to {file_path}. {len(data) - new_articles} articles already exist.')

    return f'{new_articles} new articles saved to {file_path}. {len(data) - new_articles} articles already exist.'



def execute(ticker):
    news_articles = get_news(ticker)
    
    data_to_save = []

    
    for article in news_articles:
        logger.info(f'Processing article: {article["title"]}')
        url = article['url']
        title = article['title']
        
        content = extract_content_from_url(url)
        logger.info("Content extracted.")
        
        if content:  # Sometimes, the content might be empty or None
            sentiment_score = get_sentiment_score(content)
            logger.info(f'Sentiment score: {sentiment_score}')
            date = article['publishedAt'].split('T')[0]
            data_to_save.append([date, ticker, sentiment_score, url, title])

    
    logger.info('Saving data...')
    return save_to_csv(data_to_save, ticker)


def main():
    # Rerun the execute function on the tickers in /news_sentiment_data/tickers.txt
    with open('news_sentiment_data/tickers.txt', 'r') as f:
        tickers = f.read().splitlines()

    data = []

    for ticker in tickers:
        data.append(execute(ticker))
    

    logger.info('Done.')

    c.print('\n\n')

    for item in data:
        logger.info(item)



def extract_content_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    paragraphs = soup.find_all('p')
    content = ' '.join([p.text for p in paragraphs])

    return content


if __name__ == '__main__':
    main()