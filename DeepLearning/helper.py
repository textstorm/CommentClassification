
import tensorflow as tf
import numpy as np
import collections
import utils
import time

from tensorflow.python.ops import lookup_ops
from model import TextCNN, TextRNN, Base, RNNWithAttention

class TrainModel(collections.namedtuple("TrainModel", ("graph", "model", "iterator"))):
  pass

def build_train_model(args, name="train_model", scope=None):
  vocab_dir = args.vocab_dir
  data_dir = args.train_dir
  graph = tf.Graph()

  with graph.as_default():
    vocab_table = lookup_ops.index_table_from_file(vocab_dir, default_value=0) #default_value can also be -1
    dataset = tf.data.TextLineDataset(data_dir).skip(1)

    if args.model_type == "cnn":
      iterator = utils.get_iterator(dataset=dataset,
                                    vocab_table=vocab_table,
                                    batch_size=args.batch_size,
                                    max_len=args.max_len,
                                    random_seed=args.random_seed,
                                    shuffle=True)
      model = TextCNN(args, iterator, name=name)
    elif args.model_type in ["rnn", "attention"]:
      iterator = utils.get_iterator(dataset=dataset,
                                    vocab_table=vocab_table,
                                    batch_size=args.batch_size,
                                    max_len=None,
                                    random_seed=args.random_seed,
                                    shuffle=True)
      if args.model_type == "rnn":
        model = TextRNN(args, iterator, name=name)
      elif args.model_type == "attention":
        model = RNNWithAttention(args, iterator, name=name)
    else:
      raise ValueError("Unknown model_type %s" % args.model_type)

  return TrainModel(graph=graph, model=model, iterator=iterator)

class EvalModel(
    collections.namedtuple("EvalModel", ("graph", "model", "iterator"))):
  pass

def build_eval_model(args, name="eval_model", scope=None):
  vocab_dir = args.vocab_dir
  data_dir = args.valid_dir
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "eval"):
    vocab_table = lookup_ops.index_table_from_file(vocab_dir, default_value=0)
    dataset = tf.data.TextLineDataset(data_dir).skip(1)

    if args.model_type == "cnn":
      iterator = utils.get_iterator(dataset=dataset,
                                    vocab_table=vocab_table,
                                    batch_size=args.max_size_cnn,
                                    max_len=args.max_len,
                                    random_seed=args.random_seed,
                                    shuffle=False)
      model = TextCNN(args, iterator, name=name)
    elif args.model_type in ["rnn", "attention"]:
      iterator = utils.get_iterator(dataset=dataset,
                                    vocab_table=vocab_table,
                                    batch_size=args.max_size_rnn,
                                    max_len=None,
                                    random_seed=args.random_seed,
                                    shuffle=False)
      if args.model_type == "rnn":
        model = TextRNN(args, iterator, name=name)
      elif args.model_type == "attention":
        model = RNNWithAttention(args, iterator, name=name)
    else:
      raise ValueError("Unknown model_type %s" % args.model_type)

  return EvalModel(graph=graph, model=model, iterator=iterator)

class TestModel(collections.namedtuple("TestModel",("graph", "model", "iterator"))):
  pass

def build_test_model(args, name="test_model", scope=None):
  vocab_dir = args.vocab_dir
  data_dir = args.test_dir
  graph = tf.Graph()

  with graph.as_default(), tf.container(scope or "test"):
    vocab_table = lookup_ops.index_table_from_file(vocab_dir, default_value=0)
    dataset = tf.data.TextLineDataset(data_dir).skip(1)

    if args.model_type == "cnn":
      iterator = utils.get_test_iterator(dataset=dataset,
                                        vocab_table=vocab_table,
                                        batch_size=args.max_size_cnn,
                                        max_len=args.max_len)
      model = TextCNN(args, iterator, name=name)
    elif args.model_type in ["rnn", "attention"]:
      iterator = utils.get_test_iterator(dataset=dataset,
                                        vocab_table=vocab_table,
                                        batch_size=args.max_size_rnn,
                                        max_len=None)
      if args.model_type == "rnn":
        model = TextRNN(args, iterator, name=name)
      elif args.model_type == "attention":
        model = RNNWithAttention(args, iterator, name=name)
    else:
      raise ValueError("Unknown model_type %s" % args.model_type)

  return TestModel(graph=graph, model=model, iterator=iterator)

def load_model(model, ckpt, session, name):
  start_time = time.time()
  model.saver.restore(session, ckpt)
  session.run(tf.tables_initializer())
  utils.print_out(
      "loaded %s model parameters from %s, time %.2fs" %
      (name, ckpt, time.time() - start_time))
  return model

def create_or_load_model(model, model_dir, session, name):
  latest_ckpt = tf.train.latest_checkpoint(model_dir)
  if latest_ckpt:
    model = load_model(model, latest_ckpt, session, name)
  else:
    start_time = time.time()
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    utils.print_out("created %s model with fresh parameters, time %.2fs" %
        (name, time.time() - start_time))

  global_step = model.global_step.eval(session=session)
  return model, global_step