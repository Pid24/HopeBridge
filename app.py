# Importing essential libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import re
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# Setup flask app
app = Flask(__name__)


# Load the model NB and vectorizer
filename = 'model/Model-RF.pkl'
classifier = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl','rb'))


@app.route('/', methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
	return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
	return render_template('contact.html')

@app.route('/article', methods=['GET'])
def article():
	return render_template('article.html')

@app.route('/hope', methods=['GET'])
def hope():
	return render_template('hope.html')


def text_preporocessing(text):
    # Hapus semua special characters
    processed_tweet = re.sub(r'\W', ' ', str(text))

    # Hapus semua single characters
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

    # Hapus single characters dari awal
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet) 

    # Substitusi multiple spaces dengan single space
    processed_tweet= re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

    # Hapus prefixed 'b'
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

    # Ubah menjadi Lowercase
    processed_tweet = processed_tweet.lower()
    return processed_tweet


@app.route('/predict-text', methods=['POST'])
def predict_text():
    if request.method == 'POST':
        input_text = request.form['input_text']
        text_prep = text_preporocessing(input_text)
        text_matrix = vectorizer.transform([text_prep])
        pred = classifier.predict(text_matrix.toarray())
        proba = classifier.predict_proba(text_matrix.toarray())

        if pred == [0 ]:
            text = 'Negatif'
            class_proba = int(proba[0][0].round(2)*100)
            
        elif pred == 1 :
            text = 'Positif'
            class_proba = int(proba[0][1].round(2)*100) 

        
        return render_template('hope.html', text_result=input_text, result_pred=text, result_proba=class_proba)
        
    

if __name__ == '__main__':
	app.run(debug=True)