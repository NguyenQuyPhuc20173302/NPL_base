from flask import Flask, render_template, url_for, flash, request, redirect, Response
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
import pickle
from sklearn.linear_model import LogisticRegression
from pyvi import ViTokenizer
import numpy as np
import gensim

app = Flask(__name__)


# def preprocessing_doc(doc):
#     lines = gensim.utils.simple_preprocess(doc)
#     lines = ' '.join(lines)
#     lines = ViTokenizer.tokenize(lines)
#
#     return lines
#
#
# classify = linear_model.LogisticRegression()
# loaded_model = pickle.load(open('model_linear.sav', 'rb'))
# tfidf_vect = pickle.load(open('tfidf_vect.pkl', 'rb'))


@app.route("/")
def home():
    return render_template('index.html', result=' ')


@app.route("/predict", methods=['POST'])
def predict():
    text = request.form['text']
    name_model = request.form['dropdown']
    # test_doc = preprocessing_doc(text)
    kq = ''
    if name_model == 'SVM':
        print(name_model)
    elif name_model == 'RandomForsst':
        print(name_model)
    elif name_model == 'DNN':
        print(name_model)
    elif name_model == 'LSTM':
        print(name_model)
    elif name_model == 'BRNN':
        print(name_model)
    # else:
    #     test_doc_tfidf = tfidf_vect.transform([test_doc])
    #     kq = loaded_model.predict(test_doc_tfidf)[0]

    return render_template('index.html', result=kq)


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
