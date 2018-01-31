
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import time

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

train = train.fillna("unknown")
test = test.fillna("unknown")

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train_mes, train_l = train['comment_text'], train[], test_size=0.2, random_state=2)
word_vect = word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1,1),
    max_features=20000)
char_vect = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=20000)

word_vect.fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
char_vect.fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
comments_train_word = word_vect.transform(train_mes)
comments_train_char = char_vect.transform(train_mes)
comments_valid_word = word_vect.transform(valid_mes)
comments_valid_char = char_vect.transform(valid_mes)
comments_test_word = word_vect.transform(test['comment_text'])
comments_test_char = char_vect.transform(test['comment_text'])
comments_train = hstack((comments_train_word,comments_train_char))
comments_valid = hstack((comments_valid_word,comments_valid_char))
comments_test = hstack((comments_test_word,comments_test_char))

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))
model = LogisticRegression()

for i, j in enumerate(col):
    print('fit '+j)
    t0 = time.time()
    model.fit(comments_train, train_l[j], comments_valid, valid_l[j])
    print("%.2f secs ==> [%d/6]LogisticRegression().fit()" % (time.time()-t0, i+1))
    preds[:, i] = model.predict_proba(comments_test)[:, 1]
    gc.collect()

subm = pd.read_csv('../data/sample_submission.csv')    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('submission_lrv2.csv', index=False)
