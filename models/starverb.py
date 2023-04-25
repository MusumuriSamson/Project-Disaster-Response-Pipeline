from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import sent_tokenize
from nltk import pos_tag
import pandas as pd
from train_classifier import tokenize

'''

The StartVerbExtractor class consists of function starting_verb function 
tokenizes the input text into sentences using 
the nltk library, and then tags the first word in each phrase using part-of-speech 
tagging. 

The method returns True if the first word is a verb (as indicated by a 
part-of-speech tag of 'VB' or 'VBP') or if it is the term 'RT' 
(which may suggest a retweet). 

If not, it returns False.

'''


class StartVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        '''
        Text is converted into sentences using the nltk package, and the 
        first word in each phrase is tagged using part-of-speech tagging.

        '''

        sentence_list = sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        '''
        Because this transformer does not require any training, the fit method 
        has no effect.

        '''

        return self

    def transform(self, X):
        '''
        The transform method runs the starting_verb method on each element in 
        the input Series and returns a DataFrame containing the binary feature.

        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
