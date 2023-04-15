import sys
import pandas as pd
import warnings
import sqlite3
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

import pickle
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    :param file_path: takes in the filepath of the database to read.
    :return: the function returns the table as a dataframe.

    """

    conn = sqlite3.connect(database_filepath)

    df = pd.read_sql_query("SELECT * from Disaster_data", con=conn)

    X = df['message']
    # Y = df['genre']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    conn.close()

    return X, Y, category_names


def tokenize(text):
    """
    :param text: takes in a string value and using nltk methods normalizes,tokenizes and lemmatizes it.


    :return: list of nlp processed text.

    """

    # Find all urls if any exists in the text and replace it with the word 'url_placeholder'
    url_format = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    all_urls = re.findall(url_format, text)

    for u in all_urls:
        text = text.replace(u, 'url_placeholder')

    # Tokenize the text
    tokenized_text = word_tokenize(text.lower())

    # Lemmatize the text
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for token in tokenized_text:
        lemmatized_text = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(lemmatized_text)

    return clean_tokens


def build_model():
    """
        With feature extraction methods from scikit learn this function builds a pipeline with a classifier.

    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            RandomForestClassifier(min_samples_split=2, n_estimators=10)))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model: tT.
    :param X_test: takes in a set of input features.
    :param Y_test: takes in a set of input labels.
    :param category_names: takes in a list of values containing column names

    :return:

    """

    y_pred = model.predict(X_test)

    accuracy_scores = []
    precision_scores = []
    f1_scores = []
    all_recall = []

    for i, cat in enumerate(category_names):

        accuracy_scores.append(accuracy_score(
            Y_test.values[:, i], y_pred[:, i])*100)
        precision_scores.append(precision_score(
            Y_test.values[:, i], y_pred[:, i], average='weighted')*100)
        f1_scores.append(
            f1_score(Y_test.values[:, i], y_pred[:, i], average='weighted')*100)
        all_recall.append(recall_score(
            Y_test.values[:, i], y_pred[:, i], average='weighted')*100)

    all_scores_dict = dict(zip(category_names, zip(
        accuracy_scores, precision_scores, f1_scores, all_recall)))

    all_scores_df = pd.DataFrame(all_scores_dict).T

    all_scores_df.columns = ['Accuracy', 'Precision', 'F1', 'Recall']

    all_scores_df = all_scores_df.reset_index().rename(
        columns={'index': 'Feature'})

    return all_scores_df


def save_model(model, model_filepath):
    """
    :param model: our classification estimator
    :param model_filepath: location of the path to store our model

    :return:

    """

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
