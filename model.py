
import tensorflow as tf
import numpy as np

class Base(object):
  def __init__(self, args, iterator, name=None):
    self.sentence_length = args.sentence_length
    self.nb_classes = args.nb_classes
    self.vocab_size = args.vocab_size
    self.embed_size = args.embed_size
    self.max_grad_norm = args.max_grad_norm

    self.iterator = iterator
    self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    self.batch_size = tf.size(self.iterator.sentence_length)
    self.learning_rate = args.learning_rate
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)

    self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]),
                                 trainable=True, name="embedding")
    self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size])
    self.embedding_init = self.embedding.assign(self.embedding_placeholder)
    self.embed_inp = tf.nn.embedding_lookup(self.embedding, self.iterator.comments)

  def _weight_variable(self, shape, name, initializer=None):
    initializer = tf.truncated_normal_initializer(stddev=0.1)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def _build_embedding(self, vocab_size, embed_size, name):
    with tf.variable_scope("embedding") as scope:
      embedding = self._weight_variable([vocab_size, embed_size], name=name)
    return embedding

  def _bias_variable(self, shape, name, initializer=None):
    initializer = tf.constant_initializer(0.)
    if initializer:
      initializer = initializer
    return tf.get_variable(shape=shape, initializer=initializer, name=name)

  def train(self, sess, keep_prob):
    return sess.run([self.train_op, 
                     self.loss,
                     self.global_step,
                     self.batch_size,
                     self.summary], 
                     feed_dict={self.keep_prob: keep_prob})

  def test(self, sess, keep_prob):
    return sess.run([self.loss, self.batch_size], 
                    feed_dict={self.keep_prob: keep_prob})

  def get_logits(self, sess, keep_prob):
    return sess.run(self.logits, feed_dict={self.keep_prob: keep_prob})

class TextCNN(Base):
  def __init__(self, args, iterator, name=None):
    self.filter_sizes = args.filter_sizes
    self.num_filters = args.num_filters

    super(TextCNN, self).__init__(args=args, iterator=iterator, name=name)

    embed_exp = tf.expand_dims(self.embed_inp, -1)

    pooling_output = []
    for i, filter_size in enumerate(self.filter_sizes):
      with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
        weight = self._weight_variable(filter_shape, name=("weight_%d" % i))
        bias = self._bias_variable(self.num_filters, name=("bias_%d" % i))
        conv = tf.nn.conv2d(input=embed_exp,
                            filter=weight,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
        conv = tf.layers.batch_normalization(conv)
        hidden = tf.nn.relu(conv + bias)
        pool = tf.nn.max_pool(value=hidden,
                              ksize=[1, self.sentence_length - filter_size + 1, 1, 1],
                              strides=[1, 1, 1, 1],
                              padding="VALID",
                              name="pooling")
        pooling_output.append(pool)

    num_filters_total = self.num_filters * len(self.filter_sizes)
    h_pool = tf.concat(pooling_output, -1)
    self.h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.name_scope("dropout"):
      self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

    with tf.name_scope("output"):
      self.scores = tf.layers.dense(self.h_drop, self.nb_classes, name="scores")
      self.logits = tf.nn.sigmoid(self.scores)
      self.predictions = tf.argmax(self.scores, 1, name="predictions")

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=self.scores)
      self.loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables())

class TextRNN(Base):
  def __init__(self, args, iterator, name=None):
    super(TextRNN, self).__init__(args=args, iterator=iterator, name=name)
    self.hidden_size = args.hidden_size
    self.model_type = args.model_type
    self.rnn_type = args.rnn_type
    self.rnn_layers = args.rnn_layers

    with tf.variable_scope("rnn"):
      if self.rnn_type == "rnn":
        cell = self.build_rnn_cell(self.hidden_size, self.rnn_layers, self.keep_prob)
        rnn_output, rnn_state = tf.nn.dynamic_rnn(cell=cell, 
                                                  inputs=self.embed_inp, 
                                                  dtype=tf.float32,
                                                  sequence_length=self.iterator.sentence_length)
        self.rnn_state = rnn_state

      elif self.rnn_type == "bi_rnn":
        num_layers = self.rnn_layers // 2
        fw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
        bw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
        rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, 
                                                                cell_bw=bw_cell, 
                                                                inputs=self.embed_inp,
                                                                dtype=tf.float32, 
                                                                sequence_length=self.iterator.sentence_length)
        if num_layers > 1:
          rnn_state = tuple(rnn_state[0][num_bi_layers - 1], rnn_state[1][num_bi_layers - 1])
        self.rnn_state = tf.concat(rnn_state, -1)
        rnn_output = tf.concat(rnn_output, -1)
        self.rnn_output = tf.layers.max_pooling1d(rnn_output, rnn_output.get_shape().as_list()[1], 1)

    with tf.name_scope("output"):
      tmp = tf.reshape(self.rnn_output, [self.batch_size, self.hidden_size])
      pre_score = tf.layers.dense(tmp, 32, activation=tf.nn.elu, name="pre_scores")
      self.scores = tf.layers.dense(pre_score, self.nb_classes, name="scores")
      self.logits = tf.nn.sigmoid(self.scores)
      self.predictions = tf.argmax(self.scores, 1, name="predictions")

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=self.scores)
      self.loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    self.saver = tf.train.Saver(tf.global_variables())

  def single_cell(self, num_units, keep_prob):
    """ single cell """
    cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, input_keep_prob=keep_prob)
    return cell

  def build_rnn_cell(self, num_units, num_layers, keep_prob=1.0):
    cell_list = []
    for i in range(num_layers):
      cell = self.single_cell(num_units, keep_prob)
      cell_list.append(cell)

    if num_layers == 1:
      return cell_list[0]
    else:
      return tf.contrib.rnn.MultiRNNCell(cell_list)