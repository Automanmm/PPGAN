"""
Train GAN model.
"""

import os
import tensorflow as tf
import tensorflow.contrib as tc
from data import Loader
from PPGAN import Generator
from discriminator import Discriminator
# import tester


tf.app.flags.DEFINE_string("tables", "", "tables info")
tf.app.flags.DEFINE_string("buckets", "", "buckets info")
tf.app.flags.DEFINE_string("outputs", "", "odps outputs info")

tf.app.flags.DEFINE_string('data_click0_table', './data/click0.csv', 'Directory in which data is stored.')
tf.app.flags.DEFINE_string('data_click1_table', './data/click1.csv', 'Directory in which data is stored.')
tf.app.flags.DEFINE_string('data_dir', './data', 'Directory in which data is stored.')
tf.app.flags.DEFINE_string('vocab_path', './data/vocab.csv', 'Where to save vocabulary.')

tf.app.flags.DEFINE_string('save_dir', './models', 'Where to save checkpoint models.')
tf.app.flags.DEFINE_integer('pn_batch_size', 100, 'Batch size of pointer networks.')
tf.app.flags.DEFINE_integer('ctr_batch_size', 100, 'Batch size of ctr networks.')
tf.app.flags.DEFINE_integer('seq_length', 20, 'The default lengths of long title sequences.')
tf.app.flags.DEFINE_integer('train_epochs', 1000, 'Number of epochs to run.')
tf.app.flags.DEFINE_integer('G_pretrain_epochs', 10, 'The epochs of Generator pre_training.')
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for Adam optimizer.')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5).')
tf.app.flags.DEFINE_integer('d_iters', 5, 'The iterations of Discriminator training each epoch.')
tf.app.flags.DEFINE_integer('g_iters', 5, 'The iterations of Generation training each epoch.')

FLAGS = tf.app.flags.FLAGS


def pretrain_G(sess, saver, generator, trainSet, epochs, pre_train_step):
    # pre_train Generator
    print('Start pre_training generator...')
    # tempData = {'output_prob': [], 'x_batch': [], 'u_batch': [], 'x_lengths': [], 'y_hot_batch': []}
    for i in range(epochs):
        G_loss0, G_acc0 = 0, 0
        for itr in range(trainSet.n_batches):
            x_batch, u_batch, x_lengths, y_batch, y_hot_batch = trainSet.next_pn_batch()
            train_dict = {generator.encoder_inputs: x_batch,
                          generator.user_inputs: u_batch,
                          generator.input_lengths: x_lengths,
                          generator.pointer_labels: y_batch,
                          generator.pointer_hot_labels: y_hot_batch}
            loss, _, output_prob, pre_labels = sess.run([generator.loss,
                                                         pre_train_step,
                                                         generator.pointer_prob,
                                                         generator.rank_pointers],
                                                        feed_dict=train_dict)
            jishu = 0
            for j, line in enumerate(pre_labels):
                for word in line:
                    if word in y_batch[j]:
                        jishu = jishu + 1
            acc = jishu*1.0 / (FLAGS.pn_batch_size * 5)

            G_loss0 += loss
            G_acc0 += acc
        print("Pre_train Generator: epoch:{}, loss:{:.2f}, acc:{:.2f}".format(
            i, G_loss0 / trainSet.n_batches, G_acc0 / trainSet.n_batches))

    saver.save(sess, os.path.join(FLAGS.save_dir, 'pre_G/pretrain_generator'))


def main(args):
    # load data
    click1_Set = Loader(os.path.join(FLAGS.data_dir, 'click1.csv'), FLAGS.vocab_path,
                        FLAGS.pn_batch_size, FLAGS.ctr_batch_size, FLAGS.seq_length)
    click0_Set = Loader(os.path.join(FLAGS.data_dir, 'click0.csv'), FLAGS.vocab_path,
                        FLAGS.pn_batch_size, FLAGS.ctr_batch_size, FLAGS.seq_length)

    # pretrain graph
    generator = Generator(click1_Set.vocab_size)
    pre_optimize = tf.train.AdamOptimizer(FLAGS.learning_rate)
    pre_train_step = pre_optimize.minimize(generator.loss, var_list=generator.vars)
    saver = tf.train.Saver()
    title_inputs = tf.placeholder(tf.int32, [None, FLAGS.seq_length], name='title_inputs')
    title_distribution = tf.placeholder(tf.float32, [None, FLAGS.seq_length], name='title_distribution')

    # formal train graph
    generator_fake = Generator(click0_Set.vocab_size)
    real_discriminator = Discriminator(click1_Set.vocab_size, title_inputs, title_distribution)
    fake_discriminator = Discriminator(click0_Set.vocab_size, generator_fake.encoder_inputs,
                                       generator_fake.pointer_prob)
    reg = tc.layers.apply_regularization(
        tc.layers.l1_regularizer(2.5e-5),
        weights_list=[var for var in tf.global_variables() if 'kernel' or 'W1' or 'W2' in var.name]
    )
    D_real_loss = tf.reduce_mean(real_discriminator.predictions)
    D_fake_loss = tf.reduce_mean(fake_discriminator.predictions)
    D_loss = D_fake_loss - D_real_loss
    D_loss_reg = D_loss + reg
    D_optimize = tf.train.RMSPropOptimizer(FLAGS.learning_rate)

    # WGAN lipschitz-penalty
    alpha = tf.random_uniform(
        shape=[tf.shape(title_distribution)[0], 1, 1],
        minval=0.,
        maxval=1.
    )
    differences = generator_fake.pointer_prob_hot - title_distribution
    interpolates = title_distribution + (alpha * differences)
    gradients = tf.gradients(Discriminator(click0_Set.vocab_size, generator_fake.encoder_inputs,
                                       interpolates).predictions, [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    D_loss_reg = D_loss + 10 * gradient_penalty

    D_train_step = D_optimize.minimize(D_loss_reg, var_list=fake_discriminator.vars)

    G_loss = - tf.reduce_mean(fake_discriminator.predictions)
    G_loss_reg = G_loss + reg
    G_optimize = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    G_train_step = G_optimize.minimize(G_loss_reg, var_list=generator_fake.vars)

    saver_G = tf.train.Saver(var_list=generator.vars)
    saver_D = tf.train.Saver(var_list=real_discriminator.vars)

    # for var in tf.global_variables():
    #     print(var.name)
    # training precess
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pretrain_G(sess, saver, generator, click1_Set, FLAGS.G_pretrain_epochs, pre_train_step)
        print('Start training...')
        for i in range(FLAGS.train_epochs):
            d_iters = FLAGS.d_iters
            g_iters = FLAGS.g_iters
            for _ in range(0, d_iters):
                x_batch_fake, u_batch_fake, x_lengths_fake, y_batch_fake, y_hot_batch_fake = click0_Set.next_pn_batch()
                x_batch_real, u_batch_real, x_lengths_real, y_batch_real, y_hot_batch_real = click1_Set.next_pn_batch()
                D_dict = {generator_fake.encoder_inputs: x_batch_fake,
                          generator_fake.user_inputs: u_batch_fake,
                          generator_fake.input_lengths: x_lengths_fake,
                          generator_fake.pointer_labels: y_batch_fake,
                          generator_fake.pointer_hot_labels: y_hot_batch_fake,
                          title_inputs: x_batch_real,
                          title_distribution: y_hot_batch_real,
                          real_discriminator.user_inputs: u_batch_real,
                          fake_discriminator.user_inputs: u_batch_fake}
                sess.run(fake_discriminator.d_clip)
                loss_Real, loss_Fake, lossD, _ = sess.run([D_real_loss, D_fake_loss, D_loss, D_train_step], feed_dict=D_dict)

            for _ in range(0, g_iters):
                x_batch_fake, u_batch_fake, x_lengths_fake, y_batch_fake, y_hot_batch_fake = click0_Set.next_pn_batch()
                x_batch_real, u_batch_real, x_lengths_real, y_batch_real, y_hot_batch_real = click1_Set.next_pn_batch()
                D_dict = {generator_fake.encoder_inputs: x_batch_fake,
                          generator_fake.user_inputs: u_batch_fake,
                          generator_fake.input_lengths: x_lengths_fake,
                          generator_fake.pointer_labels: y_batch_fake,
                          generator_fake.pointer_hot_labels: y_hot_batch_fake,
                          title_inputs: x_batch_real,
                          title_distribution: y_hot_batch_real,
                          real_discriminator.user_inputs: u_batch_real,
                          fake_discriminator.user_inputs: u_batch_fake}
                lossG, _ = sess.run([G_loss, G_train_step], feed_dict=D_dict)

            print("epoch:{}, D_loss:{:.2f}, G_loss:{:.2f}, loss_Real:{:.2f}, loss_Fake:{:.2f}, Sum_loss:{:.2f}"
                  .format(i, lossD, lossG, loss_Real, loss_Fake, lossD+lossG))

        saver_G.save(sess, os.path.join(FLAGS.save_dir, 'G/train_generator'))
        saver_D.save(sess, os.path.join(FLAGS.save_dir, 'D/train_discriminator'))

    # tester.generator_predict(FLAGS)


if __name__ == '__main__':
    tf.app.run()

