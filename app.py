import os
import logging
import json
from pathlib import Path

from flask import Flask, request, jsonify, send_file, after_this_request, make_response
from flask_cors import CORS

from fastai.vision.all import *
import google.generativeai as genai
from pymongo import MongoClient

# Import functions from utils.py (make sure utils.py is in the same folder)
from utils import (
    fetch_news, process_article, comparative_analysis, generate_tts,
    advanced_summarize, extended_analysis, build_embeddings
)
import os

# Initialize Flask app and enable CORS for all routes
app = Flask(__name__)
CORS(app)

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

# --- Configuration for Facial Analysis and Chatbot functionalities ---
# MongoDB connection for facial analysis recommendations
client = MongoClient('mongodb+srv://Sayandas:Sayanat2001@cluster0.iu2x4ch.mongodb.net/')
db = client['facialAnalysisApp']
collection = db['recommendation']

# Configure the API key for Google AI (Generative AI)
os.environ["GEMINI_API_KEY"] = "AIzaSyBllq6SnaKfsYvOgsHb2jW446LCE4ljRDw"
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Generation configuration for the generative model (chatbot)
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 212,
    "response_mime_type": "text/plain",
}

# Create a GenerativeModel instance with the specified configuration
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="Provide expert and short advice on skincare routines, recommend products based on different skin types and conditions, and answer questions with a friendly and professional tone. Keep the replies very brief and precise. Also, you can ask for details to get a better understanding of the problem.",
)

# --- Functions for Facial Analysis (Fastai) ---
def load_model():
    model_path = Path('export_fixed.pkl')  # Use Path for cross-platform compatibility
    learn = load_learner(model_path)
    return learn

def load_recommendations():
    recommendations = {}
    # Fetch all the records from the MongoDB collection
    data = collection.find()
    for item in data:
        condition = item['condition']
        products = item['products']
        recommendations[condition] = products
    return recommendations

def get_labels(learner):
    return learner.dls.vocab

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow requests from any domain
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

def predict_image(img_path, learner):
    img = PILImage.create(img_path)
    pred, pred_idx, probs = learner.predict(img)
    labels = get_labels(learner)
    predictions = {labels[i]: float(probs[i]) for i in range(len(labels))}
    return predictions

# --- Endpoints from the Facial Analysis & Chatbot App ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        learner = load_model()
        recommendations = load_recommendations()

        image_file = request.files['image']
        img_path = 'temp.jpg'
        image_file.save(img_path)

        predictions = predict_image(img_path, learner)
        os.remove(img_path)

        # Get recommended products for each condition from the predictions
        recommended_products = {condition: recommendations.get(condition, []) for condition in predictions}

        response = jsonify({'predictions': predictions, 'recommendations': recommended_products})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers['Content-Type'] = 'application/json'
        return response
    except Exception as e:
        logging.error(str(e))
        response = jsonify({'error': str(e)})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers['Content-Type'] = 'application/json'
        return response

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    try:
        # Get the user's message and conversation history from the request
        user_input = request.json.get('message')
        history = request.json.get('history', [])

        # Ensure history is a list before processing
        if not isinstance(history, list):
            history = []

        # Convert history to the format required by the generative AI SDK
        formatted_history = [
            {"role": item["role"], "parts": [item["content"]]} for item in history if "role" in item and "content" in item
        ]

        # Start a new chat session with the model, including the formatted history
        chat_session = model.start_chat(history=formatted_history)

        # Send the user's message to the model
        response_obj = chat_session.send_message(user_input)
        bot_response = response_obj.text

        # Append the conversation to the history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "model", "content": bot_response})

        return jsonify({'response': bot_response, 'history': history}), 200
    except Exception as e:
        logging.error(str(e))
        return jsonify({'error': str(e)}), 500

# --- Endpoints from the News Analysis API (api.py) ---
@app.route("/", methods=["GET"])
def home():
    print("Home endpoint accessed.")
    return {"status": "Backend is running!"}

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    company = data.get("company")
    if not company:
        return jsonify({"error": "Company name not provided"}), 400

    print(f"Analyzing company: {company}")

    # Fetch BBC articles asynchronously
    raw_articles = fetch_news(company, num_articles=15)
    if not raw_articles:
        return jsonify({"error": f"No articles found for '{company}'."}), 404

    processed_articles = [process_article(a) for a in raw_articles]

    # Minimal output
    articles_output = [{
        "Title": art["Title"],
        "Summary": art["Summary"],
        "Sentiment": art["Sentiment"],
        "Topics": art["Topics"]
    } for art in processed_articles]

    # Basic + Extended Analysis
    comp_analysis = comparative_analysis(processed_articles)
    ext_analysis = extended_analysis(processed_articles)
    embeddings = build_embeddings(processed_articles)

    # Determine majority sentiment
    sentiment_dist = comp_analysis.get("Sentiment Distribution", {})
    majority = "Neutral"
    pos_count = sentiment_dist.get("Positive", 0)
    neg_count = sentiment_dist.get("Negative", 0)
    if pos_count > neg_count:
        majority = "Positive"
    elif neg_count > pos_count:
        majority = "Negative"

    # Summarize all articles
    aggregated_text = " ".join(a["Summary"] for a in processed_articles)
    short_summary = advanced_summarize(aggregated_text, num_sentences=1)
    final_sentiment = f"{company}'s latest news is mostly {majority}. {short_summary}"
    final_sentiment = final_sentiment.replace(" .", ".")

    # Generate TTS audio
    audio_file = generate_tts(f"कंपनी {company}. {final_sentiment}", lang='hi')
    print(f"TTS generated: {audio_file}")

    return jsonify({
        "Company": company,
        "Articles": articles_output,
        "Comparative Sentiment Score": comp_analysis,
        "Extended Analysis": ext_analysis,
        "Final Sentiment Analysis": final_sentiment,
        "Embeddings": embeddings,  # for semantic search
        "Audio": audio_file
    })

@app.route('/audio/<filename>', methods=['GET'])
def get_audio(filename):
    print(f"Serving audio file: {filename}")
    @after_this_request
    def remove_file(response):
        try:
            os.remove(filename)
            print(f"Removed file: {filename}")
        except Exception as error:
            print(f"Error removing file {filename}: {error}")
        return response

    try:
        with open(filename, "rb") as f:
            audio_data = f.read()
        response = make_response(audio_data)
        response.headers["Content-Type"] = "audio/mp3"
        return response
    except Exception as e:
        print(f"Error reading audio file {filename}: {e}")
        return jsonify({"error": "Audio file not found"}), 404

# --- Run the Combined Flask App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False)