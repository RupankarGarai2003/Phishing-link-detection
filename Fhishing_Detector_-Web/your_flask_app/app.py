from flask import Flask, render_template, request
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from utils import tokenizer
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    mnb_tfidf = load('mnb_tfidf_model.joblib')

    # Load the TF-IDF vectorizer used during training
    tfidf_vectorizer = load('tvec.joblib')

    # Get user input from the form
    user_input = request.form['url']

    # Vectorize the user input using the same TF-IDF vectorizer used during training
    user_input_vectorized = tfidf_vectorizer.transform([user_input])

    # Predict using the loaded model
    prediction = mnb_tfidf.predict(user_input_vectorized)[0]

    # Display the prediction on the result page
    return render_template('result.html', prediction=prediction, url=user_input)

if __name__ == '__main__':
    app.run(debug=True)
