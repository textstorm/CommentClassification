
import tensorflow as tf
import numpy as np
import collections
import tqdm
import sys

def load_data(file_dir):
  f = open(file_dir, 'r')
  sentences = []
  while True:
    sentence = f.readline()
    if not sentence:
      break

    sentence = sentence.strip().lower()
    sentences.append(sentence)
  f.close()
  return sentences

def load_glove(pretrain_dir, vocab):
  embedding_dict = {}
  f = open(pretrain_dir,'r')
  for row in f:
    values = row.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embedding_dict[word] = vector
  f.close()
  vocab_size = len(vocab)
  embedding = np.zeros((vocab_size, 300))
  for idx, word in enumerate(vocab):
    word_vector = embedding_dict.get(word)
    if word_vector is not None:
      embedding[idx] = word_vector
  return embedding

def load_fasttext(pretrain_dir, vocab):
  embedding_dict = {}
  f = open(pretrain_dir, 'r')
  for row in tqdm.tqdm(f.read().split("\n")[1:-1]):
    values = row.split(" ")
    word = values[0]
    vector = np.array([float(num) for num in values[1:-1]])
    embedding_dict[word] = vector
  f.close()
  vocab_size = len(vocab)
  embedding = np.zeros((vocab_size, 300))
  for idx, word in enumerate(vocab):
    word_vector = embedding_dict.get(word)
    if word_vector is not None:
      embedding[idx] = word_vector
  return embedding

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "comments",
                                           "labels",
                                           "sentence_length"))):
  pass

def pad_sequences(sequence, max_len):
  seq_len = tf.size(sequence)
  sequence = tf.cond(seq_len > max_len, 
                     lambda: tf.slice(sequence, [seq_len - max_len], [max_len]),
                     lambda: tf.pad(sequence, [[0, max_len - seq_len]]))
  return sequence

def deal_rnn_sequence(sequence, max_len=500):
  seq_len = tf.size(sequence)
  sequence = tf.cond(seq_len > max_len, 
                     lambda: tf.slice(sequence, [seq_len - max_len], [max_len]),
                     lambda: sequence)
  return sequence

def deal_very_long_test_data(sequence, max_len=1000):
  seq_len = tf.size(sequence)
  sequence = tf.cond(seq_len > max_len, 
                     lambda: tf.slice(sequence, [seq_len - max_len], [max_len]),
                     lambda: sequence)
  return sequence

def read_row(csv_row):
  record_defaults = [[0], [""], [""], [0], [0], [0], [0], [0], [0]]
  row = tf.decode_csv(csv_row, record_defaults=record_defaults)
  return row[2], row[3:]

def get_iterator(dataset,
                 vocab_table,
                 batch_size,
                 max_len,
                 random_seed=123,
                 shuffle=True,
                 output_buffer_size=None):

  if not output_buffer_size:
    output_buffer_size = batch_size * 10000

  unk_id = tf.cast(vocab_table.lookup(tf.constant("<unk>")), tf.int32)

  if shuffle:
    dataset = dataset.shuffle(output_buffer_size, random_seed)

  dataset = dataset.map(lambda line: read_row(line))
  dataset = dataset.map(lambda x, y: (tf.string_split([x]).values, y))
  dataset = dataset.map(lambda x, y: (tf.cast(vocab_table.lookup(x), tf.int32), y))
  if max_len is not None: dataset = dataset.map(lambda x, y: (pad_sequences(x, max_len), y))
  else: dataset = dataset.map(lambda x, y: (deal_rnn_sequence(x), y))
  dataset = dataset.map(lambda x, y: (x, tf.cast(y, tf.float32), tf.size(x)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  #comments
            tf.TensorShape([None]), #labels
            tf.TensorShape([])), # length

        padding_values=(
            unk_id,
            0.,
            0))

  batch_dataset = batching_func(dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (comments, labels, sentence_length) = (batch_iterator.get_next())

  return BatchedInput(
    initializer=batch_iterator.initializer,
    comments=comments,
    labels=labels,
    sentence_length=sentence_length)

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto

def read_test_row(csv_row):
  record_defaults = [[0], [""], [""]]
  row = tf.decode_csv(csv_row, record_defaults=record_defaults)
  return row[2]

def get_test_iterator(dataset,
                      vocab_table,
                      batch_size,
                      max_len):

  unk_id = tf.cast(vocab_table.lookup(tf.constant("<unk>")), tf.int32)
  dataset = dataset.map(lambda line: read_test_row(line))
  dataset = dataset.map(lambda line: tf.string_split([line]).values)
  dataset = dataset.map(lambda line: tf.cast(vocab_table.lookup(line), tf.int32))
  dataset = dataset.map(lambda line: deal_very_long_test_data(line))
  if max_len is not None: dataset = dataset.map(lambda line: pad_sequences(line, max_len))
  else: dataset = dataset.map(lambda x, y: (deal_rnn_sequence(x), y))
  dataset = dataset.map(lambda line: (line, tf.cast(line, tf.float32), tf.size(line)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  #comments
            tf.TensorShape([None]),  #labels but not use
            tf.TensorShape([])), # length

        padding_values=(
            unk_id,
            0.,
            0))

  batch_dataset = batching_func(dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (comments, labels, sentence_length) = (batch_iterator.get_next())

  return BatchedInput(
    initializer=batch_iterator.initializer,
    comments=comments,
    labels=labels,
    sentence_length=sentence_length)

def print_out(s, f=None, new_line=True):
  if isinstance(s, bytes):
    s = s.decode("utf-8")

  if f:
    f.write(s.encode("utf-8"))
    if new_line:
      f.write(b"\n")

  # stdout
  out_s = s.encode("utf-8")
  if not isinstance(out_s, str):
    out_s = out_s.decode("utf-8")
  print out_s,

  if new_line:
    sys.stdout.write("\n")
  sys.stdout.flush()