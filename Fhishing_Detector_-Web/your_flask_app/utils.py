# utils.py
import re

def tokenizer(url):
    tokens = re.split(r'[\./-]', url)
    common_substrings = ['com', 'www']
    tokens = [token for token in tokens if token not in common_substrings]
    return tokens
