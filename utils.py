
import tensorflow as tf
import collections

class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "comments",
                                           "labels",
                                           "sequence_length"))):
  pass

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
  dataset = dataset.map(lambda x, y: (pad_sequences(x, max_len), y))
  dataset = dataset.map(lambda x, y: (x, y, tf.size(x)))

  def batching_func(x):
    return x.padded_batch(
        batch_size,
        padded_shapes=(
            tf.TensorShape([None]),  # src
            tf.TensorShape([None]), #tgt_in
            tf.TensorShape([])), # tgt_len

        padding_values=(
            unk_id,  # src
            unk_id,  # tgt_input
            0))  # tgt_len -- unused

  batch_dataset = batching_func(dataset)
  batch_iterator = batch_dataset.make_initializable_iterator()
  (comments, labels, sequence_length) = (batch_iterator.get_next())

  return BatchedInput(
    initializer=batch_iterator.initializer,
    comments=comments,
    labels=labels,
    sequence_length=sequence_length)