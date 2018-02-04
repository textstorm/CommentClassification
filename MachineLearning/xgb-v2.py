
import xgboost as xgb
import pandas as pd
import pickle
import numpy as np
import gc

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=400):
  param = {}
  param['objective'] = 'binary:logistic'
  param['eta'] = 0.12
  param['gamma'] = 1
  param['max_depth'] = 5
  param['silent'] = 1
  param['eval_metric'] = 'logloss'
  param['min_child_weight'] = 1
  param['subsample'] = 0.7
  param['colsample_bytree'] = 0.7
  param['seed'] = seed_val
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

if __name__ == '__main__':
  print('loading data')
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
    print 'training xgboost model for {}...'.format(col)
    model = runXGB(comments_train, labels_train[col], comments_valid, labels_valid[col])
    preds[:,i] = model.predict(xgb.DMatrix(comments_test))
    gc.collect()

  subm = pd.read_csv('../data/sample_submission.csv')    
  submid = pd.DataFrame({'id': subm["id"]})
  submission = pd.concat([submid, pd.DataFrame(preds, columns = cols)], axis=1)
  submission.to_csv('submission-xgb.csv', index=False)