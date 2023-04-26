import sys
import pandas as pd
import warnings
import sqlite3
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    :param file_path: takes in the filepath of the database to read.
    :return: the function returns the table as a dataframe.

    """

    conn = sqlite3.connect(database_filepath)

    df = pd.read_sql_query("SELECT * from Disaster_data", con=conn)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns

    conn.close()

    return X, Y, category_names


def tokenize(text):
    """
    :param text: takes in a string value and using nltk methods normalizes,
    tokenizes and lemmatizes it.

    :return: list of nlp processed text.

    """

    # Find all urls if any exists in the text and replace it with the word
    # 'url_placeholder'
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
        This function uses pipeline to feed into GridSearchCV in order to determine the best parameters.

    """

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'clf__estimator__learning_rate': [1.0, 1.5],
        'clf__estimator__n_estimators': [10, 20]}

    grid_search = GridSearchCV(
        pipeline, param_grid=parameters, cv=5, n_jobs=-1, verbose=2)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model - ML model
    :param X_test - test messages
    :param y_test - categories for test messages
    :param category_names - category name for y

    :return: Scores of (precision, recall, f1-score) for each output category of the dataset.

    """
    y_pred = model.predict(X_test)

    print(classification_report(Y_test.values,
          y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    :param model: our classification estimator
    :param model_filepath: location of the path to store our model

    :return: None

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
