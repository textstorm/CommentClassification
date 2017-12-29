
import tensorflow as tf
import collections

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

def read_row(csv_row):
  record_defaults = [[0], [0.], [""], [0], [0], [0], [0], [0], [0], [0]]
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
  dataset = dataset.map(lambda x, y: (pad_sequences(x, max_len), tf.cast(y, tf.float32)))
  dataset = dataset.map(lambda x, y: (x, y, tf.size(x)))

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
