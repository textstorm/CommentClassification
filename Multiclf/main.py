
import pandas as pd
import config
import utils
import time
import os

from sklearn.model_selection import train_test_split
from model import TextCNN, TextRNN, TextFNN
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):
  print "loadding data and labels from dataset"
  train = pd.read_csv(args.train_dir)
  x_train = train["comment_text"]
  target_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  preds = []

  x = []
  for line in x_train:
    if len(line) > 0:
      x.append(utils.review_to_wordlist(line.strip()))
  print "loaded %d comments from dataset" % len(x)
  print x[1]

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

  with tf.Graph().as_default():
    config_proto = utils.get_config_proto()
    sess = tf.Session(config=config_proto)
    for col in target_cols:
      y = train[col].tolist()
      x_train, x_eval, y_train, y_eval = train_test_split(
          x_vector, y, test_size=0.2, shuffle=True, random_state=2018, stratify=y)

      if args.model_type == "cnn":
        model = TextCNN(args, "TextCNN")
      elif args.model_type == "fnn":
        model = TextFNN(args, "TextFNN")
      elif args.model_type == "rnn":
        model = TextRNN(args, "TextRNN")      
      sess.run(tf.global_variables_initializer())

      print "training %s model for %s" % (args.model_type, col)
      for epoch in range(1, args.nb_epochs + 1):
        print "epoch %d start" % epoch, "\n", "- " * 50
        loss, total_comments = 0., 0
        if args.model_type in ["cnn", "fnn"]:
          train_batch = utils.get_batches(x_train, y_train, args.batch_size)
          test_batch = utils.get_batches(x_eval, y_eval, args.max_size_cnn)
        elif args.model_type in ["rnn"]:
          train_batch = utils.get_batches(x_train, y_train, args.batch_size, type="rnn")
          test_batch = utils.get_batches(x_eval, y_eval, args.max_size_rnn, type="rnn")

        epoch_start_time = time.time()
        step_start_time = epoch_start_time
        for idx, batch in enumerate(train_batch):
          comments, comments_length, labels = batch
          _, loss_t, global_step, batch_size = model.train(sess, 
                  comments, comments_length, labels, keep_prob)

          loss += loss_t * batch_size
          total_comments += batch_size

          if global_step % 200 == 0:
            print "epoch %d, step %d, loss %f, time %.2fs" % \
              (epoch, global_step, loss_t, time.time() - step_start_time)
            run_valid(test_batch, model, sess)
            model.saver.save(sess, os.path.join(save_dir, ("%s-model.ckpt" % col)), global_step=global_step)  
            step_start_time = time.time()

          if global_step > 400:
            break

        epoch_time = time.time() - epoch_start_time
        print "%.2f seconds in this epoch" % (epoch_time)
        print "train loss %f" % (loss / total_comments)

      # all epoch finish

def run_valid(test_data, model, sess):
  loss, total_comments = 0., 0
  for batch in test_data:
    comments, comments_length, labels = batch
    loss_t, logits_t, batch_size = model.test(sess, comments, comments_length, labels, 1.0)
    total_comments += batch_size
    loss += loss_t * batch_size
  print "loss %f in %d test comments" % (loss / total_comments, total_comments)  

# def run_test(args, model, global_step, sess, iterator):
#   sess.run(iterator.initializer)
#   total_logits = []
#   while True:
#     try:
#       logits = model.get_logits(sess, 1.).tolist()
#       total_logits += logits
#     except tf.errors.OutOfRangeError:
#       break
#   print np.array(total_logits).shape

# def test(args):
#   test = pd.read_csv(args.valid_dir)

if __name__ == '__main__':
  args = config.get_args()
  main(args)