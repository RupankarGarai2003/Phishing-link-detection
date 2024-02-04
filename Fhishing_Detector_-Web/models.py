from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PhishingResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(255), nullable=False)
    is_phishing = db.Column(db.Boolean, nullable=False)

    def __init__(self, url, is_phishing):
        self.url = url
        self.is_phishing = is_phishing
