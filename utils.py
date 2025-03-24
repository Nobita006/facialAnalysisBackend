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

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  # Windows symlink warning

import torch
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available

from transformers import pipeline
import spacy
from googletrans import Translator  # googletrans==4.0.0-rc1
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# NLTK downloads
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')

# spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    logging.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Summarizer
try:
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        revision="a4f8f3e",
        device=device
    )
except Exception as e:
    logging.error(f"Error loading summarization pipeline: {e}")
    summarizer = None

# Sentiment pipeline (binary)
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
except Exception as e:
    logging.error(f"Error loading transformer sentiment analyzer: {e}")
    sentiment_pipeline = None

# KeyBERT
try:
    kw_model = KeyBERT()
except Exception as e:
    logging.error(f"Error initializing KeyBERT: {e}")
    kw_model = None

# Translator for Hindi TTS
translator = Translator()

# SentenceTransformer for semantic search
try:
    st_model = SentenceTransformer('all-MiniLM-L6-v2', device=("cuda" if torch.cuda.is_available() else "cpu"))
except Exception as e:
    logging.error(f"Error loading SentenceTransformer: {e}")
    st_model = None

# In-memory cache
cache = {}
CACHE_EXPIRY = 300  # 5 minutes

def get_from_cache(key: str):
    if key in cache:
        entry = cache[key]
        if time.time() - entry["time"] < CACHE_EXPIRY:
            return entry["data"]
    return None

def set_cache(key: str, data):
    cache[key] = {"data": data, "time": time.time()}

async def fetch_article_content(session: aiohttp.ClientSession, url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        async with session.get(url, headers=headers, timeout=10) as response:
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            paragraphs = soup.find_all('p')
            if paragraphs:
                return " ".join(p.get_text(strip=True) for p in paragraphs[:3])
            return ""
    except Exception as e:
        logging.error(f"Error fetching article content asynchronously from {url}: {e}")
        return ""

async def process_card(card) -> Dict:
    link_tag = card.find('a', attrs={"data-testid": "internal-link"})
    link = None
    if link_tag and link_tag.has_attr('href'):
        href = link_tag['href']
        link = "https://www.bbc.com" + href if href.startswith('/') else href
    title_tag = card.find('h2', attrs={"data-testid": "card-headline"})
    title = title_tag.get_text(strip=True) if title_tag else "No Title Found"
    snippet_tag = card.find('div', class_='sc-4ea10043-3')
    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
    summary = snippet

    if link:
        async with aiohttp.ClientSession() as session:
            content = await fetch_article_content(session, link)
            if content and len(content) > len(snippet):
                summary = content
    return {
        "Title": title,
        "Link": link,
        "Summary": summary if summary else "Summary not available"
    }

async def process_page(page: int, base_url: str, headers: Dict, company: str) -> List[Dict]:
    params = {"q": company, "page": page}
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url, params=params, headers=headers, timeout=10) as response:
            text = await response.text()
            soup = BeautifulSoup(text, 'html.parser')
            result_cards = soup.find_all('div', attrs={"data-testid": "newport-card"})
            tasks = [process_card(card) for card in result_cards]
            page_articles = await asyncio.gather(*tasks)
            return page_articles

def fetch_news(company_name: str, num_articles: int = 15) -> List[Dict[str, str]]:
    cache_key = f"bbc_{company_name}_{num_articles}"
    cached = get_from_cache(cache_key)
    if cached:
        logging.info("Returning cached news data")
        return cached

    base_url = "https://www.bbc.com/search"
    headers = {"User-Agent": "Mozilla/5.0"}
    articles = []
    max_pages = 5

    async def main():
        nonlocal articles
        for page in range(1, max_pages + 1):
            page_articles = await process_page(page, base_url, headers, company_name)
            articles.extend(page_articles)
            if len(articles) >= num_articles:
                break

    asyncio.run(main())
    articles = articles[:num_articles]
    set_cache(cache_key, articles)
    return articles

def summarize_text(text: str, num_sentences: int = 3) -> str:
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences]) if sentences else text

def advanced_summarize(text: str, num_sentences: int = 1) -> str:
    """
    Summarize with a transformer pipeline if text is sufficiently long.
    """
    if summarizer and len(text.split()) > 50:
        try:
            max_len = 130 if len(text.split()) >= 130 else len(text.split())
            result = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
            return result[0]['summary_text']
        except Exception as e:
            logging.error(f"Transformer summarization error: {e}")
    return summarize_text(text, num_sentences)

def analyze_sentiment(text: str) -> Tuple[str, Dict[str, float]]:
    """
    Attempt to produce Neutral if the pipeline or fallback logic suggests it.
    """
    if sentiment_pipeline:
        try:
            result = sentiment_pipeline(text)[0]  # e.g. {"label": "POSITIVE", "score": 0.98}
            label = result.get("label", "NEUTRAL").upper()  # "POSITIVE"/"NEGATIVE"/(rarely) "NEUTRAL"
            score = result.get("score", 0.0)
            if label == "POSITIVE":
                return "Positive", {"compound": score, "raw": result}
            elif label == "NEGATIVE":
                return "Negative", {"compound": score, "raw": result}
            else:
                return "Neutral", {"compound": score, "raw": result}
        except Exception as e:
            logging.error(f"Transformer sentiment analysis error: {e}")

    # Fallback to VADER if pipeline fails
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        return "Positive", scores
    elif compound <= -0.05:
        return "Negative", scores
    else:
        return "Neutral", scores

def extract_topics(text: str, num_topics: int = 5) -> List[str]:
    """
    KeyBERT if available, else fallback to spaCy noun-chunks.
    """
    if kw_model:
        try:
            keywords = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=num_topics
            )
            return [kw[0] for kw in keywords]
        except Exception as e:
            logging.error(f"KeyBERT topic extraction error: {e}")
    doc = nlp(text)
    chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    freq = {}
    for c in chunks:
        freq[c] = freq.get(c, 0) + 1
    sorted_chunks = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in sorted_chunks[:num_topics]]

def translate_to_hindi(text: str) -> str:
    try:
        translated = translator.translate(text, dest='hi')
        return translated.text
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return text

def process_article(article: Dict[str, str]) -> Dict:
    """
    Summarize, analyze sentiment, and extract topics for a single article.
    """
    original_summary = article.get("Summary", "")
    if original_summary and original_summary != "Summary not available":
        summarized_text = advanced_summarize(original_summary)
        sentiment, scores = analyze_sentiment(summarized_text)
        topics = extract_topics(summarized_text)
    else:
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
    """
    Basic coverage differences, sentiment distribution, topic overlap.
    """
    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for art in articles:
        s = art.get("Sentiment", "Neutral")
        sentiment_counts[s] += 1

    pos_titles = [a["Title"] for a in articles if a["Sentiment"] == "Positive"]
    neg_titles = [a["Title"] for a in articles if a["Sentiment"] == "Negative"]

    comparisons = []
    if pos_titles and neg_titles:
        comparisons.append({
            "Comparison": f"Positive articles ({', '.join(pos_titles)}) highlight opportunities, while negative articles ({', '.join(neg_titles)}) emphasize challenges.",
            "Impact": "This indicates mixed market sentiment."
        })
    all_titles = [a["Title"] for a in articles]
    comparisons.append({
        "Comparison": f"Overall, the coverage spans: {', '.join(all_titles)}.",
        "Impact": "The news reflects a broad spectrum of perspectives."
    })

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
    """
    Named entity extraction and word frequency
    """
    all_text = " ".join(a["Summary"] for a in articles if a["Summary"])
    doc = nlp(all_text)
    entities = [ent.text for ent in doc.ents]
    entity_counts = Counter(entities).most_common(10)

    tokens = word_tokenize(all_text.lower())
    tokens = [t for t in tokens if t.isalpha() and len(t) > 3]
    word_counts = Counter(tokens).most_common(10)

    return {
        "Entity Counts": entity_counts,
        "Word Counts": word_counts
    }

def build_embeddings(articles: List[Dict]) -> List[List[float]]:
    """
    Build embeddings for each article's summary for semantic search
    """
    if not st_model:
        return []
    summaries = [a["Summary"] for a in articles]
    emb = st_model.encode(summaries, convert_to_tensor=True)
    return emb.cpu().numpy().tolist()

def generate_tts(text: str, lang: str = 'hi') -> str:
    """
    Hindi TTS using gTTS
    """
    hindi_text = translate_to_hindi(text)
    try:
        tts = gTTS(text=hindi_text, lang=lang)
        filename = f"tts_{uuid.uuid4().hex}.mp3"
        tts.save(filename)
        logging.info(f"TTS generated: {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error generating TTS: {e}")
        return ""
