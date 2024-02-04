import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report

from joblib import dump

def tokenizer(url):
    tokens = re.split(r'[\./-]', url)
    common_substrings = ['com', 'www']
    tokens = [token for token in tokens if token not in common_substrings]

    return tokens

url_df = pd.read_csv("Malicious URLs.csv")
feedback_url_df = pd.read_csv("feedback_url_df.csv") if os.path.exists("user_feedback.csv") else pd.DataFrame(columns=['URLs', 'Class'])
test_url = url_df["URLs"][4]
test_percentage = 0.2
train_df, test_df = train_test_split(url_df, test_size=test_percentage, random_state=42)
labels = train_df["Class"]
test_labels = test_df['Class']
print("Training Samples", len(train_df))
print("Testing Samples", len(test_df))

count_vec = CountVectorizer(analyzer=tokenizer)
count_x = count_vec.fit_transform(train_df['URLs'])

# Vectorize the training inputs with TfidfVectorizer
tVec = TfidfVectorizer(analyzer=tokenizer)
tfidf_x = tVec.fit_transform(train_df['URLs'])
dump(tVec, 'tvec.joblib')
test_count_x = count_vec.transform(test_df['URLs'])
test_tfid_x = tVec.transform(test_df['URLs'])
# Train the model with Multinomial naive Bayesian with TF-IDF
mnb_tfidf = MultinomialNB(alpha=.1)
mnb_tfidf.fit(tfidf_x, labels)
# Now test and evaluate the model
score_mnb_tfidf = mnb_tfidf.score(test_tfid_x, test_labels)
predictions_mnb_tfidf = mnb_tfidf.predict(test_tfid_x)
cmarix_mnb_tdidf = confusion_matrix(test_labels, predictions_mnb_tfidf)
classification_report_mnb_tfidf = classification_report(test_labels, predictions_mnb_tfidf)
print(classification_report_mnb_tfidf)
print(f"Accuracy: {score_mnb_tfidf}")
print("Confusion Matrix:")
print(cmarix_mnb_tdidf)
print("Classification Report:")
print(classification_report_mnb_tfidf)
from joblib import dump

# Save the model
dump(mnb_tfidf, 'mnb_tfidf_model.joblib')

# Assuming you have a separate file for the tokenizer, you need to import it here
# tokenizer.py
# def tokenizer(url):
#     # Your implementation

# Now, in malicious_link_detection_copy.py
# from tokenizer import tokenizer

# Rest of the code remains the same
