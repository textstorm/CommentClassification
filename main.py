
import os
import utils
import time
import config
import helper
import pandas as pd
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run_valid(args, valid_model, valid_sess, model_dir):
  with valid_model.graph.as_default():
    loaded_valid_model, global_step = helper.create_or_load_model(
        valid_model.model, model_dir, valid_sess, "dev")

  return _run_valid(loaded_valid_model, global_step, valid_sess, valid_model.iterator)

def _run_valid(model, global_step, sess, iterator):
  sess.run(iterator.initializer)
  total_loss, data_size = 0., 0
  while True:
    try:
      loss_t, batch_size = model.test(sess, 1.)
      total_loss += loss_t * batch_size
      data_size += batch_size
    except tf.errors.OutOfRangeError:
      break

  avg_loss = total_loss / data_size
  return avg_loss

def main(args):
  #dir
  pretrain_dir = args.glove_dir
  if args.model_type == "cnn": 
    save_dir = args.cnn_save_dir
    max_step = args.max_step_cnn
  elif args.model_type == "rnn": 
    save_dir = args.rnn_save_dir
    max_step = args.max_step_rnn

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  train_model = helper.build_train_model(args)
  valid_model = helper.build_eval_model(args)

  config_proto = utils.get_config_proto()
  train_sess = tf.Session(config=config_proto, graph=train_model.graph)
  valid_sess = tf.Session(config=config_proto, graph=valid_model.graph)

  with train_model.graph.as_default():
    loaded_train_model, global_step = helper.create_or_load_model(
        train_model.model, save_dir, train_sess, name="train")

  train_sess.run(train_model.iterator.initializer)
  # summary_writer = tf.summary.FileWriter(log_dir, train_sess.graph)

  loss = 0.
  total_reviews = 0
  epoch = 1
  epoch_start_time = time.time()
  step_start_time = epoch_start_time

  print "Epoch %d start " % (epoch)
  print "- " * 50
  vocab = utils.load_data(args.vocab_dir)
  embedding = utils.load_glove(pretrain_dir, vocab)
  train_sess.run(loaded_train_model.embedding_init, {loaded_train_model.embedding_placeholder: embedding})
  for step in range(max_step):
    try:
      _, loss_t, global_step, batch_size, summaries = loaded_train_model.train(train_sess, args.keep_prob)

      loss += loss_t * batch_size
      total_reviews += batch_size
      # summary_writer.add_summary(summaries, global_step)

      if global_step % 100 == 0:
        print "epoch %d, step %d, loss %f, time %.2fs" % \
          (epoch, global_step, loss_t, time.time() - step_start_time)   
        step_start_time = time.time()

      if global_step % 1000 == 0:
        loaded_train_model.saver.save(train_sess,
            os.path.join(save_dir, "model.ckpt"), global_step=global_step)   
        avg_loss = run_valid(args, valid_model, valid_sess, save_dir)
        print "valid loss %f after train step %d" % (avg_loss, global_step)
        step_start_time = time.time()        

    except tf.errors.OutOfRangeError:
      print "epoch %d finish, time %.2fs" % (epoch, time.time() - epoch_start_time)
      print "- " * 50
      loaded_train_model.saver.save(train_sess,
                    os.path.join(save_dir, "model.ckpt"), global_step=global_step)      
      avg_loss = run_valid(args, valid_model, valid_sess, save_dir)
      epoch_time = time.time() - epoch_start_time
      print "%.2f seconds in this epoch" % (epoch_time)
      print "train loss %f valid loss %f" % (loss / total_reviews, avg_loss)

      train_sess.run(train_model.iterator.initializer)
      epoch_start_time = time.time()
      total_reviews = 0
      loss = 0.
      epoch += 1
      print "Epoch %d start " % (epoch)
      print "- " * 50
      step_start_time = time.time()
      continue

def run_test(args, test_model, test_sess, model_dir):
  with test_model.graph.as_default():
    loaded_test_model, global_step = helper.create_or_load_model(
        test_model.model, model_dir, test_sess, "test")

  _run_test(args, loaded_test_model, global_step, test_sess, test_model.iterator)

def _run_test(args, model, global_step, sess, iterator):
  sess.run(iterator.initializer)
  total_logits = []
  while True:
    try:
      logits = model.get_logits(sess, 1.).tolist()
      total_logits += logits
    except tf.errors.OutOfRangeError:
      break
  print np.array(total_logits).shape
  write_results(np.asarray(total_logits, dtype=np.float64))

def write_results(logits):
  data = pd.read_csv(args.test_dir)
  columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  data = data.reindex(columns=columns)
  data[label_cols] = logits
  data.to_csv(args.sub_dir, index=False)

def test(args):
  if args.model_type == "cnn": 
    save_dir = args.cnn_save_dir
  elif args.model_type == "rnn": 
    save_dir = args.rnn_save_dir

  test_model = helper.build_test_model(args)
  config_proto = utils.get_config_proto()
  test_sess = tf.Session(config=config_proto, graph=test_model.graph)

  start_time = time.time()
  run_test(args, test_model, test_sess, save_dir)


if __name__ == '__main__':
  args = config.get_args()
  main(args)
  # test(args)
