"""
Generate synthetic data.
"""

import numpy as np
import random

Padding = 0
UNK = 1


class Loader(object):
    """Text data loader."""
    def __init__(self, data_path, vocab_path, pb_batch_size, ctr_batch_size, seq_length):
        self.data_path = data_path
        self.pn_batch_size = pb_batch_size
        self.ctr_batch_size = ctr_batch_size
        self.seq_length = seq_length
        self.vocab_path = vocab_path

        self.cluster_index = []
        self.item_id = []
        self.click = None
        self.long_title = []
        self.long_title_array = None
        self.short_title = []
        self.labels = None
        self.onehot_labels = None
        # self.click1_labels = None
        # self.click1_onehot_labels = None
        self.user_key_word = []
        self.user_embed = None

        self.long_embed = None
        self.short_embed = None
        self.lengths = None
        # self.long_embed_click1 = None
        # self.user_embed_click1 = None
        # self.click1_lengths = None
        self.n_ctr_batches = None
        self.n_batches = None
        self.x_distrib_ctr_batches = None
        self.x_ctr_batches, self.u_ctr_batches, self.y_ctr_batches = None, None, None
        self.x_long_title_batches = None
        self.x_batches, self.u_batches, self.x_lengths, self.y_batches, self.y_hot_batches = None, None, None, None, None
        self.pn_pointer = 0
        self.ctr_pointer = 0

        self.vocab = []
        self.vocab_size = 0
        self.wordToIndex = {}
        self.indexToWord = {}

        print('Pre-processing data...')
        self.load_vocab()
        self.pre_process()
        # self.shuffle()
        self.create_pn_batches()
        print('Pre-processed {} lines of data.'.format(self.labels.shape[0]))

    def pre_process(self):
        """Pre-process data."""
        labels_raw = []
        click_raw = []
        with open(self.data_path, 'r') as f:
            all_lines = f.readlines()
        for line in all_lines:
            item = line.strip().split(',')
            self.cluster_index.append(item[0].strip())
            self.item_id.append(item[1].strip())
            click_raw.append(int(item[2]))
            self.long_title.append(item[3].strip().split())
            self.short_title.append(item[4].strip().split())
            labels_raw.append(list(map(int, item[5].strip().split())))
            self.user_key_word.append(item[6].strip().split())
        # for i in range(len(self.long_title)):
        #     for _ in range(20-len(self.long_title[i])):
        #         self.long_title[i].append('')
        # self.long_title_array = np.array(self.long_title)

        self.click = np.array(click_raw)

        self.labels = np.array(labels_raw) - 1
        self.onehot_labels = np.zeros((len(self.long_title), self.seq_length), dtype=float)
        for i, line in enumerate(self.labels):
            self.onehot_labels[i, line] = 0.2

        self.user_embed = np.zeros((len(self.user_key_word), len(self.user_key_word[0])), dtype=int)
        for i, sample in enumerate(self.user_key_word):
            for j, word in enumerate(sample):
                if word not in self.wordToIndex:
                    self.user_embed[i][j] = self.wordToIndex['UNK']
                else:
                    self.user_embed[i][j] = self.wordToIndex[word]

        self.long_embed = np.zeros((len(self.long_title), self.seq_length), dtype=int)
        self.lengths = np.zeros(len(self.long_title), dtype=int)
        for i, sample in enumerate(self.long_title):
            self.lengths[i] = len(sample)
            for j, word in enumerate(sample):
                if j >= self.seq_length:
                    break
                elif word not in self.wordToIndex:
                    self.long_embed[i][j] = self.wordToIndex['UNK']
                else:
                    self.long_embed[i][j] = self.wordToIndex[word]
        for i in range(len(self.long_title)):
            for _ in range(20 - len(self.long_title[i])):
                self.long_title[i].append('')
        self.long_title_array = np.array(self.long_title)

    def load_vocab(self):
        self.vocab.append('Padding')
        self.vocab.append('UNK')
        with open(self.vocab_path, 'r') as f:
            all_lines = f.readlines()
        for line in all_lines:
            self.vocab.append(line.strip().split(',')[0])
        self.vocab_size = len(self.vocab)
        self.wordToIndex = dict(zip(self.vocab, list(range(len(self.vocab)))))
        self.indexToWord = dict(zip(list(range(len(self.vocab))), self.vocab))

    def shuffle(self):
        index = [i for i in range(len(self.long_embed))]
        random.shuffle(index)
        self.long_embed = self.long_embed[index]
        self.user_embed = self.user_embed[index]
        self.onehot_labels = self.onehot_labels[index]
        self.lengths = self.lengths[index]
        self.labels = self.labels[index]
        self.click = self.click[index]

    # def shuffleMerge(self):
    #     index = range(len(self.long_embed))
    #     random.shuffle(index)
    #     self.long_embed = self.long_embed[index]
    #     self.user_embed = self.user_embed[index]
    #     self.onehot_labels = self.onehot_labels[index]
    #     self.lengths = self.lengths[index]
    #     self.click = self.click[index]

    def create_pn_batches(self):
        self.n_batches = int(self.long_embed.shape[0] / self.pn_batch_size)
        # truncate training data so it is equally divisible into batches
        self.long_embed = self.long_embed[:self.n_batches * self.pn_batch_size, :]
        self.user_embed = self.user_embed[:self.n_batches * self.pn_batch_size, :]
        self.lengths = self.lengths[:self.n_batches * self.pn_batch_size]
        self.labels = self.labels[:self.n_batches * self.pn_batch_size, :]
        self.onehot_labels = self.onehot_labels[:self.n_batches * self.pn_batch_size, :]
        self.long_title_array = self.long_title_array[:self.n_batches * self.pn_batch_size, :]

        # split training data into equal sized batches
        self.x_batches = np.split(self.long_embed, self.n_batches, 0)
        self.u_batches = np.split(self.user_embed, self.n_batches, 0)
        self.x_lengths = np.split(self.lengths, self.n_batches)
        self.y_batches = np.split(self.labels, self.n_batches, 0)
        self.y_hot_batches = np.split(self.onehot_labels, self.n_batches, 0)
        self.x_long_title_batches = np.split(self.long_title_array, self.n_batches, 0)

    # def create_ctr_batches(self):
    #     self.n_ctr_batches = int(self.long_embed.shape[0] / self.ctr_batch_size)
    #     # truncate training data so it is equally divisible into batches
    #     self.long_embed = self.long_embed[:self.n_ctr_batches * self.ctr_batch_size, :]
    #     self.onehot_labels = self.onehot_labels[:self.n_ctr_batches * self.ctr_batch_size, :]
    #     self.user_embed = self.user_embed[:self.n_ctr_batches * self.ctr_batch_size, :]
    #     self.click = self.click[:self.n_ctr_batches * self.ctr_batch_size]
    #     # split training data into equal sized batches
    #     self.x_ctr_batches = np.split(self.long_embed, self.n_ctr_batches, 0)
    #     self.x_distrib_ctr_batches = np.split(self.onehot_labels, self.n_ctr_batches, 0)
    #     self.u_ctr_batches = np.split(self.user_embed, self.n_ctr_batches, 0)
    #     self.y_ctr_batches = np.split(self.click, self.n_ctr_batches)

    def next_pn_batch(self):
        """Return current batch, increment pointer by 1 (modulo n_batches)"""
        x_raw, x, u, x_len, y, y_hot = self.x_long_title_batches[self.pn_pointer], self.x_batches[self.pn_pointer], \
                                       self.u_batches[self.pn_pointer], \
                                       self.x_lengths[self.pn_pointer], self.y_batches[self.pn_pointer], \
                                       self.y_hot_batches[self.pn_pointer]
        self.pn_pointer = (self.pn_pointer + 1) % self.n_batches
        return x_raw, x, u, x_len, y, y_hot

    # def next_ctr_batch(self):
    #     """Return current batch, increment pointer by 1 (modulo n_batches)"""
    #     x, x_distrib, u, y = self.x_ctr_batches[self.ctr_pointer],\
    #               self.x_distrib_ctr_batches[self.ctr_pointer],\
    #               self.u_ctr_batches[self.ctr_pointer],\
    #               self.y_ctr_batches[self.ctr_pointer]
    #     self.ctr_pointer = (self.ctr_pointer + 1) % self.n_ctr_batches
    #     return x, x_distrib, u, y

    # def merge(self, tempData):
    #     self.long_embed = np.concatenate((self.long_embed, tempData['x_batch']))
    #     self.lengths = np.concatenate((self.lengths, tempData['x_lengths']))
    #     self.onehot_labels = np.concatenate((self.onehot_labels, tempData['output_prob']))
    #     self.user_embed = np.concatenate((self.user_embed, tempData['u_batch']))
    #     self.click = self.click - 0.1  # one side label smoothing.
    #     self.click = np.concatenate((self.click, np.zeros(len(tempData['x_batch']), dtype=int)))
    #     self.shuffleMerge()
    #     self.create_ctr_batches()
    #     print("finish generate fate corpus")


if __name__ == '__main__':
    data = Loader('data/train_1.csv', 'data/vocab_gan.csv', 100, 100, 20, 0.1)
    print (data.long_embed[:10])
    print (data.wordToIndex['Padding'],data.wordToIndex['UNK'])

