
import os
import utils
import time
import config
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import lookup_ops
from model import TextCNN, TextRNN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(args):

  data_dir = args.train_dir
  vocab_dir = args.vocab_dir
  save_dir = args.save_dir
  log_dir = args.log_dir
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  with tf.Graph().as_default():
    vocab_table = lookup_ops.index_table_from_file(vocab_dir, default_value=0)
    dataset = tf.data.TextLineDataset(data_dir)

    iterator = utils.get_iterator(
        dataset=dataset,
        vocab_table=vocab_table,
        batch_size=args.batch_size,
        max_len=args.sentence_length,
        random_seed=args.random_seed,
        shuffle=True)

    config_proto = utils.get_config_proto()
    sess = tf.Session(config=config_proto)
    if args.model_type == "cnn":
      model = TextCNN(args, iterator, "TextCNN")
      # test_batch = utils.get_batches(test_x, test_y, args.max_size)
    elif args.model_type in ["rnn", "bi_rnn"]:
      model = TextRNN(args, iterator, "TextRNN")
      # test_batch = utils.get_batches(test_x, test_y, args.max_size, type="rnn")

    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    for epoch in range(1, args.nb_epochs + 1):
      print "epoch %d start" % epoch
      print "- " * 50

      loss = 0.
      total_reviews = 0
      accuracy = 0.

      epoch_start_time = time.time()
      step_start_time = epoch_start_time
      for idx, batch in enumerate(train_batch):
        _, loss_t, accuracy_t, global_step, batch_size, summaries = model.train(sess, args.keep_prob)

        loss += loss_t * batch_size
        total_reviews += batch_size
        accuracy += accuracy_t * batch_size
        summary_writer.add_summary(summaries, global_step)

        if global_step % 50 == 0:
          print "epoch %d, step %d, loss %f, accuracy %.4f, time %.2fs" % \
            (epoch, global_step, loss_t, accuracy_t, time.time() - step_start_time)
          step_start_time = time.time()

      epoch_time = time.time() - epoch_start_time
      print "%.2f seconds in this epoch" % (epoch_time)
      print "train loss %f, train accuracy %.4f" % (loss / total_reviews, accuracy / total_reviews)

      # total_reviews = 0
      # accuracy = 0.
      # for batch in test_batch:
      #   reviews, reviews_length, labels = batch
      #   accuracy_t, batch_size = model.test(sess, reviews, reviews_length, labels, 1.0)
      #   total_reviews += batch_size
      #   accuracy += accuracy_t * batch_size
      # print "accuracy %.4f in %d test reviews" % (accuracy / total_reviews, total_reviews)

if __name__ == '__main__':
  args = config.get_args()
  main(args)