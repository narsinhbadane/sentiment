from flask import *
import numpy as np
import pandas as pd
import nltk
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Bidirectional
from keras.utils.np_utils import to_categorical
import pickle
import joblib
import re #regular expression
import nltk #natural language tool kit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
all_stopwords = stopwords.words('english') # taking only english words
del all_stopwords[142:180] # delete not realted words
del all_stopwords[116:119]

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods = ['POST'])
def text():  
    data = request.form['text1']
    df = pd.DataFrame([data], columns=['review'])
    cvfile = pickle.load(open('model_final.pkl','rb'))
    corpus = []
    for i in range(0, len(df)):
        review = re.sub('[^a-zA-Z]', ' ', df['review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    x_fresh = cvfile.transform(corpus).toarray()
    model = joblib.load('sentiment_model_final')

    y_pred = model.predict(x_fresh)

    if y_pred > 0.6:
        text1 = "Positive"
        return render_template('index.html', txt1=text1,data_val=data)
    else:
        text2 = "Negative"
        return render_template('index.html', txt2=text2,data_val=data)

if __name__ == '__main__':
    app.run()
