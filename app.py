import os
import logging
import time
import uuid
import asyncio
import aiohttp
import requests
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gtts import gTTS
from typing import List, Dict, Tuple
from collections import Counter

print("Setting HF_HUB_DISABLE_SYMLINKS_WARNING to 1")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Windows symlink warning

import torch
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
print(f"PyTorch device set: {'GPU' if device==0 else 'CPU'}")

from transformers import pipeline
import spacy
from googletrans import Translator  # googletrans==4.0.0-rc1
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
print("Logging is set up.")

# NLTK downloads
print("Downloading NLTK data...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# spaCy
try:
    print("Loading spaCy model 'en_core_web_sm'...")
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("spaCy model not found. Downloading 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Summarizer
try:
    print("Loading summarizer pipeline...")
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        revision="a4f8f3e",
        device=device
    )
except Exception as e:
    print(f"Error loading summarizer pipeline: {e}")
    logging.error(f"Error loading summarization pipeline: {e}")
    summarizer = None

# Sentiment pipeline (binary)
try:
    print("Loading sentiment analysis pipeline...")
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
except Exception as e:
    print(f"Error loading sentiment analysis pipeline: {e}")
    logging.error(f"Error loading transformer sentiment analyzer: {e}")
    sentiment_pipeline = None

# KeyBERT
try:
    print("Initializing KeyBERT model...")
    kw_model = KeyBERT()
except Exception as e:
    print(f"Error initializing KeyBERT: {e}")
    logging.error(f"Error initializing KeyBERT: {e}")
    kw_model = None

# Translator for Hindi TTS
print("Initializing translator for Hindi TTS...")
translator = Translator()

# SentenceTransformer for semantic search
try:
    print("Loading SentenceTransformer model for semantic search...")
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=("cuda" if torch.cuda.is_available() else "cpu"))
except Exception as e:
    print(f"Error loading SentenceTransformer: {e}")
    logging.error(f"Error loading SentenceTransformer: {e}")
    st_model = None

# In-memory cache
cache = {}
CACHE_EXPIRY = 300  # 5 minutes
print("In-memory cache initialized.")

def get_from_cache(key: str):
    if key in cache:
        entry = cache[key]
        if time.time() - entry["time"] < CACHE_EXPIRY:
            print(f"Cache hit for key: {key}")
            return entry["data"]
        else:
            print(f"Cache expired for key: {key}")
    else:
        print(f"No cache entry for key: {key}")
    return None

def set_cache(key: str, data):
    cache[key] = {"data": data, "time": time.time()}
    # print(f"Cache set for key: {key}")

async def fetch_article_content(session: aiohttp.ClientSession, url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    # print(f"Fetching article content from URL: {url}")
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')
            if paragraphs:
                content = " ".join(p.get_text(strip=True) for p in paragraphs[:3])
                # print(f"Fetched article content (first 100 chars): {content[:100]}...")
                return content
            print("No paragraphs found in article.")
            return ""
    except Exception as e:
        print(f"Error fetching article content from {url}: {e}")
        logging.error(f"Error fetching article content asynchronously from {url}: {e}")
        return ""

async def process_card(card) -> Dict:
    # print("Processing a result card...")
    link_tag = card.find('a', attrs={"data-testid": "internal-link"})
    link = None
    if link_tag and link_tag.has_attr('href'):
        href = link_tag['href']
        link = "https://www.bbc.com" + href if href.startswith('/') else href
        # print(f"Extracted link: {link}")
    title_tag = card.find('h2', attrs={"data-testid": "card-headline"})
    title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
    # print(f"Extracted title: {title}")
    snippet_tag = card.find('div', class_='sc-4ea10043-3')
    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
    summary = snippet

    if link:
        # print("Fetching extended content for the card...")
        async with aiohttp.ClientSession() as session:
            content = await fetch_article_content(session, link)
            if content and len(content) > len(snippet):
                summary = content
                # print("Using extended article content as summary.")
    return {
        "Title": title,
        "Link": link,
        "Summary": summary if summary else "Summary not available"
    }

async def process_page(page: int, base_url: str, headers: Dict, company: str) -> List[Dict]:
    print(f"Processing page {page} for company: {company}")
    params = {"q": company, "page": page}
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params, headers=headers, timeout=10) as response:
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            result_cards = soup.find_all('div', attrs={"data-testid": "newport-card"})
            print(f"Found {len(result_cards)} result cards on page {page}.")
            tasks = [process_card(card) for card in result_cards]
            page_articles = await asyncio.gather(*tasks)
            print(f"Processed {len(page_articles)} articles on page {page}.")
            return page_articles

def fetch_news(company_name: str, num_articles: int = 15) -> List[Dict[str, str]]:
    print(f"Fetching news for company: {company_name}, targeting {num_articles} articles.")
    cache_key = f"bbc_{company_name}_{num_articles}"
    cached = get_from_cache(cache_key)
    if cached:
        print("Returning cached news data.")
        return cached

    base_url = "https://www.bbc.com/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []
    max_pages = 5

    async def main():
        nonlocal articles
        for page in range(1, max_pages + 1):
            print(f"Fetching page {page}...")
            page_articles = await process_page(page, base_url, headers, company_name)
            articles.extend(page_articles)
            print(f"Total articles collected so far: {len(articles)}")
            if len(articles) >= num_articles:
                break

    asyncio.run(main())
    articles = articles[:num_articles]
    set_cache(cache_key, articles)
    print(f"Returning {len(articles)} articles for company: {company_name}.")
    return articles

def summarize_text(text: str, num_sentences: int = 3) -> str:
    # print("Summarizing text using basic sentence tokenization.")
    sentences = sent_tokenize(text)
    summary = " ".join(sentences[:num_sentences]) if sentences else text
    # print(f"Summary: {summary}")
    return summary

def advanced_summarize(text: str, num_sentences: int = 1) -> str:
    # print("Attempting advanced summarization...")
    if summarizer and len(text.split()) > 50:
        try:
            max_len = 130 if len(text.split()) >= 130 else len(text.split())
            result = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
            summary = result[0]['summary_text']
            # print("Advanced summarization successful.")
            return summary
        except Exception as e:
            # print(f"Transformer summarization error: {e}")
            logging.error(f"Transformer summarization error: {e}")
    # print("Falling back to basic summarization.")
    return summarize_text(text, num_sentences)

def analyze_sentiment(text: str) -> Tuple[str, Dict[str, float]]:
    # print("Analyzing sentiment for given text...")
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text)[0]  # e.g. {"label": "POSITIVE", "score": 0.98}
            label = result.get("label", "NEUTRAL").upper()  # "POSITIVE"/"NEGATIVE"/(rarely) "NEUTRAL"
            score = result.get("score", 0.0)
            # print(f"Sentiment pipeline result: {label} with score {score}")
            if label == "POSITIVE":
                return "Positive", {"compound": score, "raw": result}
            elif label == "NEGATIVE":
                return "Negative", {"compound": score, "raw": result}
            else:
                return "Neutral", {"compound": score, "raw": result}
        except Exception as e:
            print(f"Error in sentiment analysis using pipeline: {e}")
            logging.error(f"Transformer sentiment analysis error: {e}")

    print("Falling back to VADER sentiment analysis.")
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    print(f"VADER sentiment scores: {scores}")
    if compound >= 0.05:
        return "Positive", scores
    elif compound <= -0.05:
        return "Negative", scores
    else:
        return "Neutral", scores

def extract_topics(text: str, num_topics: int = 5) -> List[str]:
    # print("Extracting topics from text...")
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=num_topics
            )
            topics = [kw[0] for kw in keywords]
            # print(f"Extracted topics using KeyBERT: {topics}")
            return topics
        except Exception as e:
            print(f"Error extracting topics using KeyBERT: {e}")
            logging.error(f"KeyBERT topic extraction error: {e}")
    print("Falling back to spaCy noun-chunks for topic extraction.")
    doc = nlp(text)
    chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    freq = {}
    for c in chunks:
        freq[c] = freq.get(c, 0) + 1
    sorted_chunks = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    topics = [c for c, _ in sorted_chunks[:num_topics]]
    print(f"Extracted topics using noun-chunks: {topics}")
    return topics

def translate_to_hindi(text: str) -> str:
    print("Translating text to Hindi...")
    try:
        translated = translator.translate(text, dest='hi')
        # print(f"Translation successful (first 100 chars): {translated.text[:100]}...")
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        logging.error(f"Translation error: {e}")
        return text

def process_article(article: Dict[str, str]) -> Dict:
    # print("Processing an article...")
    original_summary = article.get("Summary", "")
    if original_summary and original_summary != "Summary not available":
        # print("Article has valid summary. Running advanced summarization, sentiment analysis, and topic extraction.")
        summarized_text = advanced_summarize(original_summary)
        sentiment, scores = analyze_sentiment(summarized_text)
        topics = extract_topics(summarized_text)
    else:
        print("Article does not have a valid summary. Skipping analysis.")
        summarized_text = original_summary
        sentiment, scores = "Neutral", {"compound": 0.0}
        topics = []
    return {
        "Title": article.get("Title") or "No Title Found",
        "Summary": summarized_text,
        "Sentiment": sentiment,
        "Sentiment Scores": scores,
        "Topics": topics
    }

def comparative_analysis(articles: List[Dict]) -> Dict:
    # print("Performing comparative analysis on articles...")
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for art in articles:
        s = art.get("Sentiment", "Neutral")
        sentiment_counts[s] += 1

    pos_titles = [a["Title"] for a in articles if a["Sentiment"] == "Positive"]
    neg_titles = [a["Title"] for a in articles if a["Sentiment"] == "Negative"]

    comparisons = []
    if pos_titles and neg_titles:
        comparison_text = f"Positive articles ({', '.join(pos_titles)}) highlight opportunities, while negative articles ({', '.join(neg_titles)}) emphasize challenges."
        # print("Adding comparative analysis with both positive and negative articles.")
        comparisons.append({
            "Comparison": comparison_text,
            "Impact": "This indicates mixed market sentiment."
        })
    all_titles = [a["Title"] for a in articles]
    overall_comparison = f"Overall, the coverage spans: {', '.join(all_titles)}."
    comparisons.append({
        "Comparison": overall_comparison,
        "Impact": "The news reflects a broad spectrum of perspectives."
    })
    print("Comparative analysis done.")

    pos_topics = set().union(*(set(a["Topics"]) for a in articles if a["Sentiment"] == "Positive"))
    neg_topics = set().union(*(set(a["Topics"]) for a in articles if a["Sentiment"] == "Negative"))
    common_topics = list(pos_topics.intersection(neg_topics))
    unique_pos = list(pos_topics - set(common_topics))
    unique_neg = list(neg_topics - set(common_topics))

    return {
        "Sentiment Distribution": sentiment_counts,
        "Coverage Differences": comparisons,
        "Topic Overlap": {
            "Common Topics": common_topics,
            "Unique Topics in Positive Articles": unique_pos,
            "Unique Topics in Negative Articles": unique_neg
        }
    }

def extended_analysis(articles: List[Dict]) -> Dict:
    # print("Performing extended analysis on articles...")
    all_text = " ".join(a["Summary"] for a in articles if a["Summary"])
    doc = nlp(all_text)
    entities = [ent.text for ent in doc.ents]
    entity_counts = Counter(entities).most_common(10)
    print("Entity counts extracted.")

    tokens = word_tokenize(all_text.lower())
    tokens = [t for t in tokens if t.isalpha() and len(t) > 3]
    word_counts = Counter(tokens).most_common(10)
    print("Word counts extracted.")

    return {
        "Entity Counts": entity_counts,
        "Word Counts": word_counts
    }

def build_embeddings(articles: List[Dict]) -> List[List[float]]:
    print("Building embeddings for semantic search...")
    if not st_model:
        print("SentenceTransformer model not available. Returning empty embeddings.")
        return []
    summaries = [a["Summary"] for a in articles]
    emb = st_model.encode(summaries, convert_to_tensor=True)
    print("Embeddings built successfully.")
    return emb.cpu().numpy().tolist()

def generate_tts(text: str, lang: str = 'hi') -> str:
    # print("Generating TTS audio...")
    hindi_text = translate_to_hindi(text)
    try:
        tts = gTTS(text=hindi_text, lang=lang)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        # print(f"TTS generated and saved as: {filename}")
        logging.info(f"TTS generated: {filename}")
        return filename
    except Exception as e:
        print(f"Error generating TTS: {e}")
        logging.error(f"Error generating TTS: {e}")
        return ""
