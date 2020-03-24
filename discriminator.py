import tensorflow as tf

# can be edited (to anything larger than vocab size) if encoding of vocab already uses 0, 1
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

# can be edited (to anything larger than vocab size) if encoding of vocab already uses 0, 1
# FLAGS = tf.app.flags.FLAGS


def leaky_relu(x, alpha=0.1):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


class Discriminator(object):

    def __init__(self, vocab_size, encoder_inputs, pointer_prob, seq_length=20,
                 cell=tf.contrib.rnn.GRUCell, n_layers=3, n_units=50, embedding_size=100,
                 fc_hidden_size=512):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        with tf.variable_scope('D_inputs', reuse=tf.AUTO_REUSE):
            self.title_inputs = encoder_inputs
            # self.title_distribution = pointer_prob
            self.title_distribution = pointer_prob + tf.random_normal(shape=tf.shape(pointer_prob), mean=0.0, stddev=0.1, dtype=tf.float32)
            self.user_inputs = tf.placeholder(tf.int32, [None, 20], name='user_inputs')

        with tf.variable_scope('D_embeddings', reuse=tf.AUTO_REUSE):
            self.word_matrix = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1.0, 1.0),
                                           trainable=True)
            self.title_embeds = tf.nn.embedding_lookup(self.word_matrix, self.title_inputs)
            self.title_distrib = tf.tile(tf.expand_dims(self.title_distribution, 2), [1, 1, embedding_size])  #(batch_size * title_seq_length)
            self.title_embeds = tf.multiply(self.title_embeds, self.title_distrib)

            self.user_embeds = tf.nn.embedding_lookup(self.word_matrix, self.user_inputs)

        with tf.variable_scope('D_conv', reuse=tf.AUTO_REUSE):
            x = tf.concat([self.title_embeds, self.user_embeds], axis=1)
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 40, 100, 1])
            conv1 = tc.layers.convolution2d(
                x, 64, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1)
            conv2 = tc.layers.convolution2d(
                conv1, 128, [4, 4], [2, 2],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(tc.layers.batch_norm(conv2))
            conv2 = tcl.flatten(conv2)
            fc1 = tc.layers.fully_connected(
                conv2, 1024,
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.identity
            )
            fc1 = leaky_relu(tc.layers.batch_norm(fc1))
            fc2 = tc.layers.fully_connected(fc1, 1, activation_fn=tf.identity)
            self.predictions = fc2
        # with tf.variable_scope('D_title_encoder', reuse=tf.AUTO_REUSE):
        #     if n_layers > 1:
        #         enc_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
        #     else:
        #         enc_cell = cell(n_units)
        #     self.encoder_outputs, _ = tf.nn.dynamic_rnn(enc_cell, self.title_embeds, dtype=tf.float32)
        #
        # with tf.variable_scope('D_title_encoder', reuse=tf.AUTO_REUSE):
        #     if n_layers > 1:
        #         u_enc_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
        #     else:
        #         u_enc_cell = cell(n_units)
        #     self.user_outputs, _ = tf.nn.dynamic_rnn(u_enc_cell, self.user_embeds, dtype=tf.float32)
        #
        #     self.encoder_outputs = tf.concat([self.encoder_outputs, self.user_outputs], axis=2)
        #
        # self.lstm_out = tf.reduce_mean(self.encoder_outputs, axis=1)
        #
        # with tf.variable_scope("D_dropout", reuse=tf.AUTO_REUSE):
        #     self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        #     self.hDrop = tf.nn.dropout(self.lstm_out, self.dropout_keep_prob)
        #
        # # Fully Connected Layer
        # with tf.variable_scope("D_fc", reuse=tf.AUTO_REUSE):
        #     W1 = tf.Variable(tf.truncated_normal(shape=[n_units * 2, fc_hidden_size],
        #                                          stddev=0.1, dtype=tf.float32), name="W1")
        #     b1 = tf.Variable(tf.constant(value=0.1, shape=[fc_hidden_size], dtype=tf.float32), name="b1")
        #     self.fc1 = tf.nn.xw_plus_b(self.hDrop, W1, b1, name="fc1")
        #     self.fc1 = tf.nn.leaky_relu(
        #         self.fc1,
        #         alpha=0.2,
        #         name=None
        #     )
        #     W2 = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, 1],
        #                                          stddev=0.1, dtype=tf.float32), name="W2")
        #     b2 = tf.Variable(tf.constant(value=0.1, shape=[1], dtype=tf.float32), name="b2")
        #     self.predictions0 = tf.nn.xw_plus_b(self.fc1, W2, b2)
        #     self.predictions = tf.squeeze(self.predictions0, 1, name='predictions')

        self.d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.vars]

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'D_' in var.name]



