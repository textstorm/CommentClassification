
import tensorflow as tf
import numpy as np

class Base(object):
  def __init__(self, args, name=None):
    self.max_len = args.max_len
    self.nb_classes = args.nb_classes
    self.vocab_size = args.vocab_size
    self.embed_size = args.embed_size
    self.max_grad_norm = args.max_grad_norm
    self.cell_type = args.cell_type

    self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
    self.input_y = tf.placeholder(tf.float32, [None, None], name='input_y')
    self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
    self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    self.is_train = tf.placeholder(tf.bool, name="istrain")

    self.batch_size = tf.shape(self.input_y)[0]
    self.learning_rate = args.learning_rate
    self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False)

#     self.embedding = self._build_embedding(self.vocab_size, self.embed_size, "encoder_embedding")
#     self.embed_inp = tf.nn.embedding_lookup(self.embedding, self.input_x)

    self.embedding = tf.Variable(tf.constant(0.0, shape=[self.vocab_size, self.embed_size]),
                                trainable=True, name="embedding")
    self.embedding_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.embed_size])
    self.embedding_init = self.embedding.assign(self.embedding_placeholder)
    self.embed_inp = tf.nn.embedding_lookup(self.embedding, self.input_x)

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

  def train(self, sess, input_x, sequence_length, input_y, keep_prob):
    return sess.run([self.train_op, 
                     self.loss,
                     self.global_step,
                     self.batch_size], 
                     feed_dict={self.input_x: input_x, self.input_y: input_y, self.keep_prob: keep_prob, 
                                self.sequence_length: sequence_length, self.is_train: True})

  def test(self, sess, input_x, sequence_length, input_y, keep_prob):
    return sess.run([self.loss, 
                     self.logits, 
                     self.batch_size], 
                     feed_dict={self.input_x: input_x, self.input_y: input_y, 
                               self.keep_prob: keep_prob, self.sequence_length: sequence_length})

  def get_logits(self, sess, input_x, sequence_length, keep_prob):
    return sess.run(self.logits, feed_dict={self.input_x: input_x, 
        self.sequence_length: sequence_length, self.keep_prob: keep_prob})

  def single_cell(self, num_units, keep_prob):
    """ single cell """
    cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    if self.cell_type == "lstm":
      cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
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

class TextCNN(Base):
  def __init__(self, args, name=None):
    self.filter_sizes = args.filter_sizes
    self.num_filters = args.num_filters

    super(TextCNN, self).__init__(args=args, name=name)

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
        hidden = tf.nn.relu(conv + bias)
        pool = tf.nn.max_pool(value=hidden,
                              ksize=[1, self.max_len - filter_size + 1, 1, 1],
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

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
      self.loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

class TextRNN(Base):
  def __init__(self, args, name=None):
    super(TextRNN, self).__init__(args=args, name=name)
    self.hidden_size = args.hidden_size
    self.model_type = args.model_type
    self.rnn_layers = args.rnn_layers

    with tf.variable_scope("rnn"):
      num_layers = self.rnn_layers // 2
      fw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
      bw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
      rnn_input = tf.layers.dropout(self.embed_inp, 0.7)
      rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, 
                                                              cell_bw=bw_cell, 
                                                              inputs=rnn_input,
                                                              dtype=tf.float32,
                                                              sequence_length=self.sequence_length)
      if num_layers > 1:
        rnn_state = tuple(rnn_state[0][num_bi_layers - 1], rnn_state[1][num_bi_layers - 1])
      self.rnn_state = tf.concat(rnn_state, -1)
      rnn_output = tf.concat(rnn_output, -1)
      self.rnn_output = tf.layers.max_pooling1d(rnn_output, self.max_len, 1)

    with tf.name_scope("output"):
      pre_score = tf.layers.dense(self.rnn_state, 32, activation=tf.nn.elu, name="pre_scores")
      self.scores = tf.layers.dense(pre_score, self.nb_classes, name="scores")
      self.logits = tf.nn.sigmoid(self.scores)

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
      self.loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

class RNNWithAttention(Base):
  def __init__(self, args, iterator, name=None):
    super(RNNWithAttention, self).__init__(args=args, name=name)
    self.hidden_size = args.hidden_size
    self.rnn_layers = args.rnn_layers
    self.attention_size = args.attention_size

    with tf.variable_scope("rnn"):
      num_layers = self.rnn_layers // 2
      fw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
      bw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
      rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                              cell_bw=bw_cell, 
                                                              inputs=self.embed_inp,
                                                              dtype=tf.float32,
                                                              sequence_length=self.sequence_length)

      rnn_output = tf.concat(rnn_output, -1)
      self.attention_output, alphas = self.attention(rnn_output, self.attention_size, return_alphas=True)

    with tf.name_scope("output"):
      W = tf.Variable(tf.truncated_normal([self.hidden_size * 2, 1], stddev=0.1))
      b = tf.Variable(tf.constant(0., shape=[1]))
      self.scores = tf.nn.xw_plus_b(self.attention_output, W, b)
      # self.scores = tf.layers.dense(self.attention_output, self.nb_classes, name="scores")
      self.logits = tf.nn.sigmoid(self.scores)
      print self.logits.get_shape().as_list()
      self.predictions = tf.argmax(self.scores, 1, name="predictions")

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.iterator.labels, logits=self.scores)
      self.loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

  def attention(self, inputs, size, return_alphas=False):
    attention_context_vector = tf.get_variable(name='attention_context_vector', shape=[size], dtype=tf.float32)
    input_projection = layers.fully_connected(inputs, size, activation_fn=tf.tanh)
    vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
    attention_weights = tf.nn.softmax(vector_attn, dim=1)
    weighted_projection = tf.multiply(inputs, attention_weights)
    outputs = tf.reduce_sum(weighted_projection, axis=1)
    if not return_alphas:
      return outputs
    else:
      return outputs, attention_weights

  def attention1(self, inputs, attention_size, return_alphas=False):
    if isinstance(inputs, tuple):
      inputs = tf.concat(inputs, 2)

    hidden_size = inputs.shape[2].value
    W_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    v = tf.tanh(tf.tensordot(inputs, W_omega, axes=1) + b_omega)

    vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
    alphas = tf.nn.softmax(vu)              # (B,T) shape also

    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)
    if not return_alphas:
      return output
    else:
      return output, alphas

class TextRNNChar(Base):
  def __init__(self, args, name=None):
    super(TextRNNChar, self).__init__(args=args, name=name)
    self.hidden_size = args.hidden_size
    self.model_type = args.model_type
    self.rnn_layers = args.rnn_layers
    self.char_embed_size = args.char_embed_size
    self.char_vocab_size = args.char_vocab_size
    self.char_filter_size = args.char_filter_size
    self.char_num_filters = args.char_num_filters

    self.char_x = tf.placeholder(tf.int32, [None, None, None], name='char_x')

    self.embedding_char = tf.get_variable("embedding_char", [self.char_vocab_size, self.char_embed_size], 
                                      dtype=tf.float32)
    self.embed_inp_char = tf.nn.embedding_lookup(self.embedding_char, self.char_x, name="embedded_input_char")
    print self.embed_inp_char.get_shape().as_list()

    with tf.variable_scope("rnn_char"):
      filter_shape = [1, self.char_filter_size, self.char_embed_size, 100]
      weight = self._weight_variable(filter_shape, name=("char_weight"))
      bias = self._bias_variable(self.char_num_filters, name=("char_bias"))
      conv = tf.nn.conv2d(input=self.embed_inp_char,
                          filter=weight,
                          strides=[1, 1, 1, 1],
                          padding="VALID",
                          name="chconv")
      char_feature = tf.reduce_max(tf.nn.relu(conv + bias), 2) 

      num_layers = self.rnn_layers // 2
      fw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
      bw_cell = self.build_rnn_cell(self.hidden_size, num_layers, self.keep_prob)
      rnn_input = tf.layers.dropout(self.embed_inp, 0.7)
      rnn_input = tf.concat([rnn_input, char_feature], -1)
      rnn_output, rnn_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, 
                                                              cell_bw=bw_cell, 
                                                              inputs=rnn_input,
                                                              dtype=tf.float32,
                                                              sequence_length=self.sequence_length)
      if num_layers > 1:
        rnn_state = tuple(rnn_state[0][num_bi_layers - 1], rnn_state[1][num_bi_layers - 1])
      self.rnn_state = tf.concat(rnn_state, -1)
      rnn_output = tf.concat(rnn_output, -1)
      self.rnn_output = tf.layers.max_pooling1d(rnn_output, self.max_len, 1)

    with tf.name_scope("output"):
      pre_score = tf.layers.dense(self.rnn_state, 32, activation=tf.nn.elu, name="pre_scores")
      self.scores = tf.layers.dense(pre_score, self.nb_classes, name="scores")
      self.logits = tf.nn.sigmoid(self.scores)

    with tf.name_scope("loss"):
      losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
      self.loss = tf.reduce_mean(losses)

    with tf.name_scope('train'):
      grads_and_vars = self.optimizer.compute_gradients(self.loss)
      grads_and_vars = [(tf.clip_by_norm(g, self.max_grad_norm), v) for g, v in grads_and_vars]
      self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    self.tvars = tf.trainable_variables()
    self.saver = tf.train.Saver(tf.global_variables())

  def train(self, sess, input_x, sequence_length, char_x, input_y, keep_prob):
    return sess.run([self.train_op, 
                     self.loss,
                     self.global_step,
                     self.batch_size], 
                     feed_dict={self.input_x: input_x, self.input_y: input_y, self.keep_prob: keep_prob, 
                                self.sequence_length: sequence_length, self.char_x: char_x, self.is_train: True})

  def test(self, sess, input_x, sequence_length, char_x, input_y, keep_prob):
    return sess.run([self.loss, 
                     self.logits, 
                     self.batch_size], 
                     feed_dict={self.input_x: input_x, self.input_y: input_y, self.keep_prob: keep_prob, 
                                self.sequence_length: sequence_length, self.char_x: char_x,self.is_train: True})

  def get_logits(self, sess, input_x, sequence_length, char_x, keep_prob):
    return sess.run(self.logits, feed_dict={self.input_x: input_x, self.char_x:char_x,
        self.sequence_length: sequence_length, self.keep_prob: keep_prob})