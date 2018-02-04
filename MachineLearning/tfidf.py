
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

print 'loading data...'
train = pd.read_csv('../data/train-tfidf.csv').fillna("unknown")
test = pd.read_csv('../data/test-tfidf.csv').fillna("unknown")
print 'train shape: {0}'.format(train.shape)
print 'test shape: {0}'.format(test.shape)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_mes, valid_mes, train_l, valid_l = train_test_split(
    train['comment_text'], train[label_cols], test_size=0.2, random_state=2018)

word_tfidf = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='word', 
    token_pattern=r'\w{1,}', ngram_range=(1, 3), lowercase=True, max_features=20000)

char_tfidf = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode', analyzer='char',
    ngram_range=(1, 5), lowercase=True, max_features=20000)

word_tfidf.fit(pd.concat([train['comment_text'], test['comment_text']], axis=0))
char_tfidf.fit(pd.concat([train['comment_text'], test['comment_text']], axis=0))
comments_train_word = word_tfidf.transform(train_mes)
comments_train_char = char_tfidf.transform(train_mes)
comments_valid_word = word_tfidf.transform(valid_mes)
comments_valid_char = char_tfidf.transform(valid_mes)
comments_test_word = word_tfidf.transform(test['comment_text'])
comments_test_char = char_tfidf.transform(test['comment_text'])
comments_train = hstack((comments_train_word, comments_train_char))
comments_valid = hstack((comments_valid_word, comments_valid_char))
comments_test = hstack((comments_test_word, comments_test_char))

print comments_train_word[:10]
print comments_train_char[:10]

pickle.dump(comments_train, open("data/train.pkl", 'wb'))
pickle.dump(comments_valid, open("data/valid.pkl", 'wb'))
pickle.dump(comments_test, open("data/test.pkl", 'wb'))
train_l.to_csv("data/train_l.csv")
valid_l.to_csv("data/valid_l.csv")