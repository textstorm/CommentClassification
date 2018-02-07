
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
import pandas as pd
import pickle
import numpy as np
import time
import gc

if __name__ == '__main__':
  print 'loading data' 
  labels_train = pd.read_csv("data/train_l.csv")
  labels_valid = pd.read_csv("data/valid_l.csv")
  print 'train labels shape: {0}'.format(labels_train.shape)
  print 'valid labels shape: {0}'.format(labels_valid.shape)

  comments_train = pickle.load(open("data/train.pkl", 'rb'))
  comments_valid = pickle.load(open("data/valid.pkl", 'rb'))
  comments_test = pickle.load(open("data/test.pkl", 'rb'))

  print 'train shape: {0}'.format(comments_train.shape)
  print 'valid shape: {0}'.format(comments_valid.shape)
  print 'test shape: {0}'.format(comments_test.shape)

  cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  preds = np.zeros((comments_test.shape[0], len(cols)))

  for i, col in enumerate(cols):
    print 'training LogisticRegression model for {}...'.format(col)
    t0 = time.time()
    model = LogisticRegression(C=10.0, solver='sag')
    model.fit(comments_train, labels_train[col])

    print "evaluating..."
    y_pred = model.predict_proba(comments_valid)[:, 1]
    auc = roc_auc_score(labels_valid[col], y_pred)
    logloss = log_loss(labels_valid[col], y_pred)
    print 'auc: {}, logloss: {}'.format(auc, logloss)

    print "%.2f secs ==> [%d/6]LogisticRegression().fit()" % (time.time() - t0, i + 1)
    preds[:, i] = model.predict_proba(comments_test)[:, 1]
    gc.collect()

  subm = pd.read_csv('../data/sample_submission.csv')    
  submid = pd.DataFrame({'id': subm["id"]})
  submission = pd.concat([submid, pd.DataFrame(preds, columns=cols)], axis=1)
  submission.to_csv('submission-lr.csv', index=False)
 