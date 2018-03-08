
import os
import utils
import time
import config
import helper
import shutil
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
  total_logits = []
  total_labels = []
  while True:
    try:
      loss_t, logits_t, labels_t, batch_size = model.test(sess, 1.)
      total_loss += loss_t * batch_size
      total_logits += logits_t.tolist()
      total_labels += labels_t.tolist()
      data_size += batch_size
    except tf.errors.OutOfRangeError:
      break

  avg_loss = total_loss / data_size
  return avg_loss

def main(args):
  #dir
  pretrain_dir = args.wordvec_dir
  save_dir = os.path.join(args.save_dir, args.model_type)
  if args.model_type == "cnn": 
    max_step = args.max_step_cnn
    keep_prob = args.keep_prob_cnn
  elif args.model_type in ["rnn", "attention"]: 
    max_step = args.max_step_rnn
    keep_prob = args.keep_prob_rnn

  test_prob = []
  for k in range(args.nfolds):
    frac = 1.0 / args.nfolds * (args.nfolds - 1)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    train_data = pd.read_csv("../data/pre/train.csv")
    data_train = train_data.sample(frac=0.9)
    data_test = train_data.drop(data_train.index)
    data_train.to_csv(args.train_dir, index=False)
    data_test.to_csv(args.valid_dir, index=False)
    train_model = helper.build_train_model(args)
    valid_model = helper.build_eval_model(args)

    config_proto = utils.get_config_proto()
    train_sess = tf.Session(config=config_proto, graph=train_model.graph)
    valid_sess = tf.Session(config=config_proto, graph=valid_model.graph)

    with train_model.graph.as_default():
      loaded_train_model, global_step = helper.create_or_load_model(
          train_model.model, save_dir, train_sess, name="train")

    train_sess.run(train_model.iterator.initializer)
    epoch = 1
    loss, total_reviews = 0., 0
    epoch_start_time = time.time()
    step_start_time = epoch_start_time

    print "Word vector use %s" % (args.wordvec_dir.split("/")[-1])
    print "Model type is %s" % (args.model_type)
    print "Epoch %d start " % (epoch)
    print "- " * 50
    vocab = utils.load_data(args.vocab_dir)
    embedding = utils.load_fasttext(pretrain_dir, vocab)
    train_sess.run(loaded_train_model.embedding_init, {loaded_train_model.embedding_placeholder: embedding})
    for line in loaded_train_model.tvars:
      print line

    for step in range(max_step):
      try:
        _, loss_t, global_step, batch_size = loaded_train_model.train(train_sess, keep_prob)

        loss += loss_t * batch_size
        total_reviews += batch_size

        if global_step % 100 == 0:
          print "epoch %d, step %d, loss %f, time %.2fs" % \
            (epoch, global_step, loss_t, time.time() - step_start_time)   
          step_start_time = time.time()

        if global_step % 100 == 0:
          loaded_train_model.saver.save(train_sess, os.path.join(save_dir, "model.ckpt"), global_step=global_step)   
          avg_loss = run_valid(args, valid_model, valid_sess, save_dir)
          print "valid loss %f after train step %d" % (avg_loss, global_step)
          step_start_time = time.time()        

      except tf.errors.OutOfRangeError:
        print "epoch %d finish, time %.2fs" % (epoch, time.time() - epoch_start_time)
        print "- " * 50
        epoch_time = time.time() - epoch_start_time
        print "%.2f seconds in this epoch" % (epoch_time)
        print "train loss %f in this epoch" % (loss / total_reviews)
        train_sess.run(train_model.iterator.initializer)
        epoch_start_time = time.time()
        loss, total_reviews = 0., 0
        epoch += 1
        print "Epoch %d start " % (epoch)
        print "- " * 50
        step_start_time = time.time()
        continue
    test_prob.append(test(args))
    shutil.rmtree(save_dir) 
  preds = np.zeros((test_prob[0].shape[0], 6))
  for prob in test_prob:
    preds += prob
    print prob[0]
  preds /= len(test_prob)
  print len(test_prob)
  write_results(preds)

def run_test(args, test_model, test_sess, model_dir):
  with test_model.graph.as_default():
    loaded_test_model, global_step = helper.create_or_load_model(
        test_model.model, model_dir, test_sess, "test")

  # _run_test(args, loaded_test_model, global_step, test_sess, test_model.iterator)
  return _run_test(args, loaded_test_model, global_step, test_sess, test_model.iterator)

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
  # write_results(np.asarray(total_logits, dtype=np.float64))
  return np.array(total_logits)

def write_results(logits):
  data = pd.read_csv(args.test_dir)
  columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
  data = data.reindex(columns=columns)
  data[label_cols] = logits
  data.to_csv(args.sub_dir, index=False)

def test(args):
  save_dir = os.path.join(args.save_dir, args.model_type)
  test_model = helper.build_test_model(args)
  config_proto = utils.get_config_proto()
  test_sess = tf.Session(config=config_proto, graph=test_model.graph)

  start_time = time.time()
  # run_test(args, test_model, test_sess, save_dir)
  return run_test(args, test_model, test_sess, save_dir)

def valid(args):
  save_dir = os.path.join(args.save_dir, args.model_type)
  valid_model = helper.build_eval_model(args)
  config_proto = utils.get_config_proto()
  valid_sess = tf.Session(config=config_proto, graph=valid_model.graph)
  avg_loss, total_labels, total_logits = run_valid(args, valid_model, valid_sess, save_dir)
  auc = tf.metrics.auc(labels=total_labels, predictions=total_logits)
  with tf.Session(config=config_proto) as sess:
    sess.run(tf.local_variables_initializer())
    auc = sess.run(auc)
  print "valid loss %f valid auc %f" % (avg_loss, auc[1])

if __name__ == '__main__':
  args = config.get_args()
  main(args)