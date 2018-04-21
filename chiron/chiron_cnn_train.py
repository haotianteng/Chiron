#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:32:32 2017

@author: haotianteng
"""
from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf

from chiron.chiron_input import read_raw_data_sets
from chiron.cnn import getcnnlogit
from six.moves import range


class Flags():
    def __init__(self):
        self.data_dir = "/home/lee/Documents/Greg/Chiron/input_fast_folder"
        self.sequence_len = 200
        self.batch_size = 1024
        self.step_rate = 1e-3
        self.max_steps = 100000


FLAGS = Flags()


def inference(x):
    logits = getcnnlogit(x, outnum=5)
    return logits


def loss(logits, seq_len, label):
    return tf.reduce_mean(
        tf.nn.ctc_loss(label, logits, seq_len, ctc_merge_repeated=False,
                       time_major=False))


def train_step(loss):
    opt = tf.train.AdamOptimizer(FLAGS.step_rate).minimize(loss)
    return opt


def prediction(logits, seq_length, label):
    logits = tf.transpose(logits, perm=[1, 0, 2])
    predict = tf.to_int32(
        tf.nn.ctc_beam_search_decoder(logits, seq_length, merge_repeated=False)[
            0][0])
    error = tf.reduce_sum(
        tf.edit_distance(predict, label, normalize=False)) / tf.to_float(
        tf.size(label.values))
    return error


def train():
    train_ds, valid_ds = read_raw_data_sets(FLAGS.data_dir,
                                            seq_length=FLAGS.sequence_len,
                                            max_reads_num=10000)
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs, y_values, y_shape)

    logits = inference(x)
    ctc_loss = loss(logits, seq_length, y)
    opt = train_step(ctc_loss)
    error = prediction(logits, seq_length, y)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.device('/gpu:0'):
        sess = tf.Session()
        sess.run(init)
        # saver.restore(sess,tf.train.latest_checkpoint('log/cnn'))
        for i in range(FLAGS.max_steps):
            batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
            indxs, values, shape = batch_y
            loss_val, _ = sess.run([ctc_loss, opt],
                                   feed_dict={x: batch_x, seq_length: seq_len,
                                              y_indexs: indxs, y_values: values,
                                              y_shape: shape})
            if i % 10 == 0:
                error_val = sess.run(error,
                                     feed_dict={x: batch_x, seq_length: seq_len,
                                                y_indexs: indxs,
                                                y_values: values,
                                                y_shape: shape})
                print(
                    "Epoch %d, batch number %d, loss: %5.2f edit_distance: %5.2f" \
                    % (train_ds.epochs_completed, train_ds.index_in_epoch,
                       loss_val, error_val))
                saver.save(sess, "log/cnn/model.ckpt", i)
        saver.save(sess, "log/cnn/final.ckpt")


def main():
    train()


if __name__ == "__main__":
    main()
