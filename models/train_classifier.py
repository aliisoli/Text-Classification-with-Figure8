import sys
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

import nltk
from nltk import WordNetLemmatizer, word_tokenize, FreqDist
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords', 'averaged_perceptron_tagger'])

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

def load_data(database_filepath):
    ''' loads the data from the cleaned sql database,
    returns X as feature and Y as target dataframes'''
    
    # load data from database
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('Figure8', con = engine)
    df = df[df['related'] != 2]
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis = 1)
    return X,Y,Y.columns.values

def delete_urls(text):
    '''uses re to remove urls from text'''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    return text

def tokenize(text):
    ''' cleans the text by lowering the case, removing punctuation,
    tokenization and lemmatization'''
    text = delete_urls(text)
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    
    clean_tokens = []
    for tok in tokens:
        try:
            float(tok)
        except:
            lemmed = WordNetLemmatizer().lemmatize(tok).lower().strip()
            lemmed = WordNetLemmatizer().lemmatize(lemmed, pos='v')
            clean_tokens.append(lemmed)

    stop_words = list(stopwords.words('english'))
    date_stopwords = ['january','february','march','april','may','june','july','august',
                      'september','october','november','december','monday','tuesday',
                      'wednesday','thursday','friday','saturday','sunday','today',
                      'yesterday','tomorrow','now']
    
    stop_words_all = set(stop_words + date_stopwords)
    
    clean_tokens =  [w for w in clean_tokens if w not in stop_words_all]   

    return clean_tokens


def build_model():
    '''creates a machine learning pipeline and uses grid search 
    to find an optimal set of parameters'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])
    
    parameters = {
        'vect__max_features': (None, 1000),
        'classifier__estimator__n_estimators': [20, 50]
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''evaluates the model and reports f1, accuracy scores 
    and prints them for display for all categories'''
    Y_pred = pd.DataFrame(model.predict(X_test))
    Y_pred.columns = Y_test.columns
    for col in Y_pred.columns:
        print(classification_report(Y_pred[col], Y_test[col]))
        
          


def save_model(model, model_filepath):
    '''saves the model as a pickle for later use'''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()