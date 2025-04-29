import streamlit as st
import pandas as pd
import joblib


st.write("## SMS Spam Classifier App")
st.write("This app classifies SMS messages as spam or ham (not spam).")

model=joblib.load("spam_model.pkl")
vectorizer=joblib.load("vectorizer.pkl")

import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("punkt")
nltk.download("stopwords")


chat_words_map = {
    'lol': 'laughing out loud', 'lmao': 'laughing my ass off', 'rofl': 'rolling on the floor laughing',
    'roflmao': 'rolling on the floor laughing my ass off', 'brb': 'be right back', 'ttyl': 'talk to you later',
    'idk': 'i do not know', 'omg': 'oh my god', 'btw': 'by the way', 'smh': 'shaking my head',
    'imo': 'in my opinion', 'imho': 'in my humble opinion', 'fyi': 'for your information',
    'afk': 'away from keyboard', 'asap': 'as soon as possible', 'bff': 'best friends forever',
    'g2g': 'got to go', 'gtg': 'got to go', 'np': 'no problem', 'nvm': 'never mind', 'wth': 'what the hell',
    'jk': 'just kidding', 'lmk': 'let me know', 'ikr': 'i know right', 'yolo': 'you only live once',
    'omfg': 'oh my freaking god', 'ty': 'thank you', 'thx': 'thanks', 'tysm': 'thank you so much',
    'plz': 'please', 'pls': 'please', 'u': 'you', 'ur': 'your', 'cya': 'see you',
    'xoxo': 'hugs and kisses', 'wyd': 'what are you doing', 'hbu': 'how about you',
    'bday': 'birthday', 'hbd': 'happy birthday', 'sup': 'what is up', 'cuz': 'because', 'coz': 'because',
    'dm': 'direct message', 'pm': 'private message', 'ftw': 'for the win', 'gr8': 'great',
    'luv': 'love', 'msg': 'message', 'txt': 'text', 'k': 'okay', 'okie': 'okay',
    'ya': 'yes', 'nah': 'no', 'yo': 'hey', 'bae': 'baby', 'fam': 'family'
}

stop_words = set(stopwords.words("english"))


def replace_chat_words(text):
    return ' '.join([chat_words_map.get(word.lower(), word) for word in text.split()])


def remove_tags(text):
    return re.sub(r'<.*?>', '', text)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords_and_digits(text):
    text = re.sub(r'\d+', '', text)
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])


def preprocess_text(text):
    text = text.lower()
    text = replace_chat_words(text)
    text = remove_tags(text)
    text = remove_punctuation(text)
    text = remove_stopwords_and_digits(text)
    return text


message = st.text_input("Enter your message")

if st.button("Classify"):
    cleaned_message = preprocess_text(message)
    vectorized_message=vectorizer.transform([cleaned_message])
    prediction= model.predict(vectorized_message)
    label = "📛 Spam" if prediction == 1 else "✅ Not Spam"
    st.subheader(f"Prediction: {label}")