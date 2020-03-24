# -*- coding:utf-8 -*-

import tensorflow as tf
from data import Loader
import os
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from PPGAN import Generator

tf.app.flags.DEFINE_string("tables", "", "tables info")
tf.app.flags.DEFINE_string("buckets", "", "buckets info")
tf.app.flags.DEFINE_string("outputs", "", "odps outputs info")

tf.app.flags.DEFINE_string('data_click0_table', './data/click0.csv', 'Directory in which data is stored.')
tf.app.flags.DEFINE_string('data_click1_table', './data/click1.csv', 'Directory in which data is stored.')
tf.app.flags.DEFINE_string('data_dir', './data', 'Directory in which data is stored.')
tf.app.flags.DEFINE_string('vocab_path', './data/vocab.csv', 'Where to save vocabulary.')

tf.app.flags.DEFINE_string('save_dir', './models', 'Where to save checkpoint models.')
tf.app.flags.DEFINE_integer('pn_batch_size', 12, 'Batch size of pointer networks.')
tf.app.flags.DEFINE_integer('ctr_batch_size', 12, 'Batch size of pointer networks.')
tf.app.flags.DEFINE_integer('seq_length', 20, 'The default lengths of long title sequences.')
tf.app.flags.DEFINE_integer('train_epochs', 1000, 'Number of epochs to run.')
tf.app.flags.DEFINE_integer('G_pretrain_epochs', 10, 'The epochs of Generator pre_training.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for Adam optimizer.')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5).')
tf.app.flags.DEFINE_integer('d_iters', 5, 'The iterations of Discriminator training each epoch.')
tf.app.flags.DEFINE_integer('g_iters', 5, 'The iterations of Generation training each epoch.')

FLAGS = tf.app.flags.FLAGS


def main(args):
    validSet = Loader(os.path.join(FLAGS.data_dir, 'test.csv'), FLAGS.vocab_path,
                      FLAGS.pn_batch_size, FLAGS.ctr_batch_size, FLAGS.seq_length)
    G = Generator()
    G(validSet.vocab_size)

    with tf.Session() as sess:

        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('models/pre_G'))

        graph = tf.get_default_graph()
        encoder_inputs = graph.get_tensor_by_name("G_inputs/encoder_inputs:0")
        user_inputs = graph.get_tensor_by_name("G_inputs/user_inputs:0")
        input_lengths = graph.get_tensor_by_name("G_inputs/input_lengths:0")
        pointer_labels = graph.get_tensor_by_name("G_outputs/pointer_labels:0")
        pointer_hot_labels = graph.get_tensor_by_name("G_outputs/pointer_hot_labels:0")

        # loss = graph.get_tensor_by_name("G_loss/loss:0")
        pointer_prob = graph.get_tensor_by_name("G_loss/pointer_prob:0")
        rank_pointers = graph.get_tensor_by_name("G_pointers/rank_pointers:0")

        print('finish loading model!')
        # test
        G_val_acc0, G_val_loss0 = 0, 0
        for itr in range(validSet.n_batches):
            x_raw, x_batch, u_batch, x_lengths, y_batch, y_hot_batch = validSet.next_pn_batch()
            test_dict = {encoder_inputs: x_batch,
                         user_inputs: u_batch,
                         input_lengths: x_lengths,
                         pointer_labels: y_batch,
                         pointer_hot_labels: y_hot_batch}
            output_prob, pre_labels = sess.run([pointer_prob, rank_pointers], feed_dict=test_dict)
            jishu = 0
            for j, line in enumerate(pre_labels):
                # print u_batch[j]
                for word in line:
                    if word in y_batch[j]:
                        jishu = jishu + 1
            acc = jishu * 1.0 / (FLAGS.pn_batch_size * 5)
            G_val_acc0 += acc
            print (pre_labels)
            print (y_batch)
            if itr == 0:
                for i in range(FLAGS.pn_batch_size):
                    print (i)
                    origin = ''
                    predict = ''
                    for j in range(20):
                        if j in y_batch[i]:
                            origin += x_raw[i, j]
                    for j in range(20):
                        if j in pre_labels[i]:
                            predict += x_raw[i, j]
                    print (i, origin)
                    print (i, predict)

        print("Test Generator: test_acc:{:.2f}".format(G_val_acc0 / validSet.n_batches))


def genMetrics(trueY, predY, binaryPredY):
    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY)
    recall = recall_score(trueY, binaryPredY)
    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)


if __name__ == '__main__':
    tf.app.run()