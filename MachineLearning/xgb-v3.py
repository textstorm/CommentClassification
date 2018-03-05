# -*- coding: utf-8 -*-
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re, string
import time
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def pr(y_i, y, x):
  p = x[y==y_i].sum(0)
  return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y,x, c0 = 4):
  y = y.values
  r = np.log(pr(1,y,x) / pr(0,y,x))
  m = LogisticRegression(C= c0, dual=True)
  x_nb = x.multiply(r)
  return m.fit(x_nb, y), r

def multi_roc_auc_score(y_true, y_pred):
  assert y_true.shape == y_pred.shape
  columns = y_true.shape[1]
  column_losses = []
  for i in range(0, columns):
      column_losses.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
  return np.array(column_losses).mean()

def runXgb(train_X, train_y, test_X, test_y=None, feature_names=None, num_rounds=1000):
  param = {}
  param['objective'] = 'binary:logistic'
  param['eta'] = 0.095
  param['max_depth'] = 5
  param['silent'] = 1
  param['eval_metric'] = 'logloss'
  param['min_child_weight'] = 0.5
  param['subsample'] = 0.7
  param['colsample_bytree'] = 0.7
  param['seed'] = 2018
  num_rounds = num_rounds

  plst = list(param.items())
  xgtrain = xgb.DMatrix(train_X, label=train_y)

  if test_y is not None:
    xgtest = xgb.DMatrix(test_X, label=test_y)
    watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
    model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
  else:
    xgtest = xgb.DMatrix(test_X)
    model = xgb.train(plst, xgtrain, num_rounds)

  return model

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
subm = pd.read_csv('../data/sample_submission.csv')

id_train = train['id'].copy()
id_test = test['id'].copy()

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1-train[label_cols].max(axis=1)
COMMENT = 'comment_text'
train[COMMENT].fillna("unknown", inplace=True)
test[COMMENT].fillna("unknown", inplace=True)

train['total_length'] = train['comment_text'].apply(len)
train['capitals'] = train['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
train['caps_vs_length'] = train.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)
test['total_length'] = test['comment_text'].apply(len)
test['capitals'] = test['comment_text'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
test['caps_vs_length'] = test.apply(lambda row: float(row['capitals'])/float(row['total_length']),axis=1)

re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')
n = train.shape[0]

word_vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word', 
    min_df = 5,
    token_pattern=r'\w{1,}',
    ngram_range=(1, 3))

all1 = pd.concat([train[COMMENT], test[COMMENT]])
word_vectorizer.fit(all1)
xtrain1 = word_vectorizer.transform(train[COMMENT])
xtest1 = word_vectorizer.transform(test[COMMENT])

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    min_df = 3,
    ngram_range=(1, 6))

all1 = pd.concat([train[COMMENT], test[COMMENT]])
char_vectorizer.fit(all1)

xtrain2 = char_vectorizer.transform(train[COMMENT])
xtest2 = char_vectorizer.transform(test[COMMENT])

nfolds = 5
xseed = 29
cval = 4
xtrain = hstack([xtrain1, 
                 xtrain2, 
                 csr_matrix(np.reshape(train['caps_vs_length'].values, (train.shape[0], 1)))], format='csr')
xtest = hstack([xtest1,
                xtest2, 
                csr_matrix(np.reshape(test['caps_vs_length'].values, (test.shape[0], 1)))], format='csr')
ytrain = np.array(train[label_cols].copy())

skf = StratifiedKFold(n_splits= nfolds, random_state=xseed)
preds = np.zeros((xtest.shape[0], len(label_cols)))
for (lab_ind,lab) in enumerate(label_cols):   
  y = train[lab].copy()
  print('label:' + str(lab_ind))
  for (f, (train_index, test_index)) in enumerate(skf.split(xtrain, y)):
    x0, x1 = xtrain[train_index], xtrain[test_index]
    y0, y1 = y[train_index], y[test_index]    
    model = runXgb(x0, y0, x1, y1)
    preds[:,lab_ind] += model.predict(xgb.DMatrix(xtest))
preds /= nfolds

# store prfull
prfull = pd.DataFrame(preds)
prfull.columns = label_cols
prfull['id'] = id_test
prfull.to_csv('prfull.csv', index=False)

# store submission
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(prfull, columns=label_cols)], axis=1)
submission.to_csv('submission-xgb.csv', index=False)