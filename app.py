from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
from nltk.stem import WordNetLemmatizer 
import os

nltk.download('wordnet')
app= Flask(__name__)
CORS(app)
model=tf.keras.models.load_model('assests/full_model.h5')
with open('assests/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('assests/custom_stopwords.json') as f:
    stopwords= set(json.load(f))



def clean(X,custom_stopwords):
    processedText = []
    
   
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in X:
        tweet = tweet.lower()
        
  
        tweet = re.sub(urlPattern,' ',tweet)
        tweet = re.sub(userPattern,'', tweet)        
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if word not in custom_stopwords:
                if len(word)>1:
                    word = wordLemm.lemmatize(word)
                    tweetwords += word+" "
            
        processedText.append(tweetwords)
        
    return processedText

def tokenize(X):
    X = tokenizer.texts_to_sequences(X)
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=100, padding='post', truncating='post')
    return X

# The /predict endpoint only supports POST requests.
@app.route('/predict', methods=['POST'])
def get_text():
    try:
        data = request.get_json()
        # Assuming the payload contains {"text": "..."}
        userText = data.get("text", "")
        # Preprocess the text
        cleaned_text = clean([userText], stopwords)
        tokenized = tokenize(cleaned_text)
        prediction = model.predict(tokenized)
        pred_class = "positive tweet" if prediction >= 0.5 else "negative tweet"
        return jsonify({'prediction': pred_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['POST'])
def download_file():
    try:
        data = request.get_json()
        userText = data.get("text", "")
        cleaned_text = clean([userText], stopwords)
        tokenized = tokenize(cleaned_text)
        prediction = model.predict(tokenized)
        pred_class = "positive tweet" if prediction >= 0.5 else "negative tweet"

        # Create the content for the text file
        file_content = f"User Input: {userText}\nPrediction: {pred_class}\n"

        # Save the content to a temporary file
        file_path = "output.txt"
        with open(file_path, "w") as file:
            file.write(file_content)

        # Send the file as a response for download
        return send_file(file_path, as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temporary file if it exists
        if os.path.exists("output.txt"):
            os.remove("output.txt")

@app.route('/')
def index():
    # This is the Flask backend server running on http://localhost:5000.
    # Use this URL in your frontend to communicate with this server.
    return jsonify({'message': 'Backend running'})

if __name__ == '__main__':
    app.run(debug=True)
