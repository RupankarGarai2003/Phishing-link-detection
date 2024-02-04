# utils.py
import re
from your_flask_app.malicious_link_Detection_copy import predict_ink
from your_flask_app.utils import tokenizer

def tokenizer(url):
    tokens = re.split(r'[\./-]', url)
    common_substrings = ['com', 'www']
    tokens = [token for token in tokens if token not in common_substrings]
    return tokens
