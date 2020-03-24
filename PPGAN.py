"""
Implementation of a Personalized Pointer GAN.
"""

import tensorflow as tf


# can be edited (to anything larger than vocab size) if encoding of vocab already uses 0, 1
END_TOKEN = 0
START_TOKEN = 1


class Generator(object):

    def __call__(self, vocab_size, n_pointers=1, n_max=5, seq_length=20, learning_rate=0.001,
                 cell=tf.contrib.rnn.GRUCell, n_layers=3, n_units=50, embedding_size=100):
        """Creates TensorFlow graph of a pointer network.

        Args:
            n_pointers (int):      Number of pointers to generate.
            seq_length (int):      Maximum sequence length of inputs to encoder.
            learning_rate (float): Learning rate for Adam optimizer.
            cell (method):         Method to create single RNN cell.
            n_layers (int):        Number of layers in RNN (assumed to be the same for encoder & decoder).
            n_units (int):         Number of units in RNN cell (assumed to be the same for all cells).
        """

        with tf.variable_scope('G_inputs'):
            self.input_lengths = tf.placeholder(tf.int32, [None], name='input_lengths')
            self.pointer_labels = tf.placeholder(tf.int32, [None, n_max], name='pointer_labels')
            self.encoder_inputs = tf.placeholder(tf.int32, [None, seq_length], name='encoder_inputs')
            self.user_inputs = tf.placeholder(tf.int32, [None, 20], name='user_inputs')
            self.pointer_hot_labels = tf.placeholder(tf.float32, [None, seq_length], name='pointer_hot_labels')
            batch_size = tf.shape(self.encoder_inputs)[0]

        with tf.variable_scope('G_outputs'):
            start_token = tf.constant([START_TOKEN], dtype=tf.int32)
            start_tokens = tf.tile(start_token, [batch_size])
            # start_tokens = tf.constant(START_TOKEN, shape=[batch_size], dtype=tf.int32)

        with tf.variable_scope('G_embeddings'):
            self.word_matrix = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=True)
            self.input_embeds = tf.nn.embedding_lookup(self.word_matrix, self.encoder_inputs)
            self.user_embeds = tf.nn.embedding_lookup(self.word_matrix, self.user_inputs)

        with tf.variable_scope('G_title_encoder', reuse=tf.AUTO_REUSE):
            if n_layers > 1:
                enc_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
            else:
                enc_cell = cell(n_units)
            self.encoder_outputs, _ = tf.nn.dynamic_rnn(enc_cell, self.input_embeds, self.input_lengths, dtype=tf.float32)

        with tf.variable_scope('G_title_encoder', reuse=tf.AUTO_REUSE):
            if n_layers > 1:
                u_enc_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
            else:
                u_enc_cell = cell(n_units)
            self.user_outputs, _ = tf.nn.dynamic_rnn(u_enc_cell, self.user_embeds, dtype=tf.float32)

            self.encoder_outputs = tf.concat([self.encoder_outputs, self.user_outputs], axis=2)

        with tf.variable_scope('G_attention', reuse=tf.AUTO_REUSE):
            attention = tf.contrib.seq2seq.BahdanauAttention(n_units, self.encoder_outputs,
                                                             memory_sequence_length=self.input_lengths)

        with tf.variable_scope('G_decoder', reuse=tf.AUTO_REUSE):
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.word_matrix, start_tokens, END_TOKEN)
            if n_layers > 1:
                dec_cell = tf.contrib.rnn.MultiRNNCell([cell(n_units) for _ in range(n_layers)])
            else:
                dec_cell = cell(n_units)
            attn_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention, alignment_history=True)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, self.word_matrix.shape[0] - 2)
            decoder = tf.contrib.seq2seq.BasicDecoder(out_cell, helper, out_cell.zero_state(batch_size, tf.float32))
            self.decoder_outputs, dec_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=n_pointers)

        with tf.variable_scope('G_pointers', reuse=tf.AUTO_REUSE):
            # tensor of shape (# pointers, batch size, max. input sequence length)
            self.pointer_prob = tf.reshape(dec_state.alignment_history.stack(), [n_pointers, batch_size, seq_length])
            self.pointers = tf.unstack(tf.nn.top_k(self.pointer_prob, k=5).indices)
            self.rank_pointers_1 = tf.squeeze(tf.nn.top_k(self.pointers, k=5).values)
            self.rank_pointers = tf.reverse(self.rank_pointers_1, [1], name='rank_pointers')

        with tf.variable_scope('G_loss', tf.AUTO_REUSE):
            hot_labels = self.pointer_hot_labels
            self.pointer_prob = tf.squeeze(self.pointer_prob, 0, name='pointer_prob')
            # prob, id = tf.nn.top_k(self.pointer_prob, 5)
            # self.pointer_prob_hot = tf.map_fn(lambda xi: tf.map_fn(lambda yi: tf.cond(tf.less(yi, xi[1]), lambda: tf.constant(0.0), lambda: tf.constant(0.2)), xi[0]),
            #                               (self.pointer_prob, tf.reduce_min(prob, axis=1)), dtype=tf.float32)

            # loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=hot_labels, logits=self.pointer_prob)
            # self.loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), name='loss')

            # loss = tf.square(self.pointer_prob - hot_labels) + tf.abs(self.pointer_prob - hot_labels)
            # self.pointer_prob = self.pointer_prob[:, :20]
            loss = tf.square(self.pointer_prob - hot_labels)
            self.loss = tf.reduce_sum(loss, name='loss')

        return self.encoder_inputs, self.pointer_prob
            # loss = tf.abs(self.pointer_prob - hot_labels)

            # labels = self.pointer_labels
            # self.equal = tf.equal(self.rank_pointers, labels)

            # self.correct = tf.cast(self.equal, tf.float32)
            # self.exact_match = tf.reduce_sum(tf.reduce_sum(self.correct, axis=1), axis=0) / (batch_size * n_max)
            # self.all_correct = tf.cast(tf.equal(tf.reduce_sum(self.correct, axis=1), n_max), tf.float32)
            # self.exact_match = tf.reduce_mean(self.all_correct)

    @property
    def vars(self):
        return [var for var in tf.global_variables() if 'G_' in var.name]
