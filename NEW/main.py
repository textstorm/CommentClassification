
import pandas as pd
import numpy as np
import config
import utils
import time
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from model import TextCNN, TextRNN, TextFNN
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
  print "loadding data and labels from dataset"
  train = pd.read_csv(args.train_dir)
  x_train = train["comment_text"]
  target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

  x = []
  for line in x_train:
    if len(line) > 0:
      x.append(utils.review_to_wordlist(line.strip()))
  print "loaded %d comments from dataset" % len(x)
  print x[1]
  y = train[target_cols].tolist()

  index2word, word2index = utils.load_vocab(args.vocab_dir)
  print index2word[:10]
  x_vector = utils.vectorize(x, word2index, verbose=True)
  print x_vector[1]

  pretrain_dir = args.wordvec_dir
  save_dir = os.path.join(args.save_dir, args.model_type)
  if args.model_type == "cnn": 
    max_step = args.max_step_cnn
    keep_prob = args.keep_prob_cnn
  elif args.model_type in ["rnn", "attention"]: 
    max_step = args.max_step_rnn
    keep_prob = args.keep_prob_rnn

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  nfolds = args.nfolds
  skf = StratifiedKFold(n_splits=nfolds, random_state=123)

  for (f, (train_index, test_index)) in enumerate(skf.split(x_vector, y)):
    x_train, x_eval = x_vector[train_index], x_vector[test_index]
    y_train, y_eval = y[train_index], y[test_index]   
    with tf.Graph().as_default():
      config_proto = utils.get_config_proto()
      sess = tf.Session(config=config_proto)
      if args.model_type == "cnn":
        model = TextCNN(args, "TextCNN")
      elif args.model_type == "fnn":
        model = TextFNN(args, "TextFNN")
      elif args.model_type == "rnn":
        model = TextRNN(args, "TextRNN")
      elif args.model_type == "attention":
        model = RNNWithAttention(args, "Attention")
      else:
        raise ValueError("Unknown model_type %s" % args.model_type)      
      sess.run(tf.global_variables_initializer())

      embedding = utils.load_fasttext(pretrain_dir, index2word)
      sess.run(model.embedding_init, {model.embedding_placeholder: embedding})
      for line in model.tvars:
        print line

      print "training %s model for %s" % (args.model_type, col)
      for epoch in range(1, args.nb_epochs + 1):
        print "epoch %d start" % epoch, "\n", "- " * 50
        loss, total_comments = 0., 0
        if args.model_type in ["cnn", "fnn"]:
          train_batch = utils.get_batches(x_train, y_train, args.batch_size, args.max_len)
          test_batch = utils.get_batches(x_eval, y_eval, args.max_size_cnn, args.max_len)
        elif args.model_type in ["rnn"]:
          train_batch = utils.get_batches(x_train, y_train, args.batch_size, args.max_len, type="rnn")
          test_batch = utils.get_batches(x_eval, y_eval, args.max_size_rnn, type="rnn")

        epoch_start_time = time.time()
        step_start_time = epoch_start_time
        for idx, batch in enumerate(train_batch):
          comments, comments_length, labels = batch
          _, loss_t, global_step, batch_size = model.train(sess, 
                  comments, comments_length, labels, keep_prob)

          loss += loss_t * batch_size
          total_comments += batch_size

          if global_step % 1000 == 0:
            print "epoch %d, step %d, loss %f, time %.2fs" % \
              (epoch, global_step, loss_t, time.time() - step_start_time)
            run_valid(test_batch, model, sess)
            thismodel_dir = os.path.join(save_dir, col)
            if not os.path.exists(thismodel_dir):
              os.makedirs(thismodel_dir)
            model.saver.save(sess, os.path.join(thismodel_dir, ("%s-model.ckpt" % col)), 
                global_step=global_step)  
            step_start_time = time.time()

          if global_step > 30000:
            break

        if global_step > 30000:
          break

        epoch_time = time.time() - epoch_start_time
        print "%.2f seconds in this epoch" % (epoch_time)
        print "train loss %f" % (loss / total_comments)

def run_valid(test_data, model, sess):
  total_logits = []
  total_labels = []
  for batch in test_data:
    comments, comments_length, labels = batch
    loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, labels, 1.0)
    total_logits += logits_t.tolist()
    total_labels += labels
  auc = roc_auc_score(np.array(total_labels), np.array(total_logits))
  print "auc %f in valid comments" % auc  

def test(args):
  save_dir = os.path.join(args.save_dir, args.model_type)
  test = pd.read_csv(args.test_dir)
  x_test = test["comment_text"]
  target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

  x = []
  for line in x_test:
    if len(line) > 0:
      x.append(utils.review_to_wordlist(line.strip()))
  print "loaded %d comments from dataset" % len(x)

  index2word, word2index = utils.load_vocab(args.vocab_dir)
  print index2word[:10]
  x_vector = utils.vectorize(x, word2index, verbose=True)
  print x_vector[1]

  target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  if args.model_type in ["cnn", "fnn"]:
    test_batch = utils.get_test_batches(x_vector, args.max_size_cnn, args.max_len)
  elif args.model_type in ["rnn"]:
    test_batch = utils.get_test_batches(x_vector, args.max_size_rnn, type="rnn")

  preds = np.zeros((test.shape[0], len(target_cols)))
  for i, col in enumerate(target_cols):
    with tf.Graph().as_default():
      config_proto = utils.get_config_proto()
      sess = tf.Session(config=config_proto)

      if args.model_type == "cnn":
        model = TextCNN(args, "TextCNN")
      elif args.model_type == "fnn":
        model = TextFNN(args, "TextFNN")
      elif args.model_type == "rnn":
        model = TextRNN(args, "TextRNN")      
      sess.run(tf.global_variables_initializer())

      model_dir = os.path.join(save_dir, col)
      latest_ckpt = tf.train.latest_checkpoint(model_dir)
      loaded_model = utils.load_model(model, latest_ckpt, sess, name="test")

      total_logits = []
      for idx, batch in enumerate(test_batch):
        comments, comments_length = batch
        logits = loaded_model.get_logits(sess, comments, comments_length, 1.).tolist()
        total_logits += logits
    
      preds[:,i] = np.array(total_logits)[:,0]

  print preds.shape
  write_results(preds)

def write_results(logits):
  cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  submission = pd.read_csv('data/sample_submission.csv')    
  submid = pd.DataFrame({'id': submission["id"]})
  submission = pd.concat([submid, pd.DataFrame(logits, columns=cols)], axis=1)
  submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
  args = config.get_args()
  main(args)
  test(args)
