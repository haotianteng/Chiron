#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Mon Mar 27 14:04:57 2017
# This module is going to be deprecated, use chiron_train and chiron_queue_input instead.
# from rnn import rnn_layers
from __future__ import absolute_import
from __future__ import print_function
import argparse
import sys
import os
import time

from distutils.dir_util import copy_tree

import tensorflow as tf
import chiron.chiron_model as model

from chiron.chiron_input import read_raw_data_sets
from chiron.chiron_input import read_tfrecord
from chiron.cnn import getcnnfeature
from chiron.cnn import getcnnlogit
from six.moves import range


def save_model():
    copy_tree(os.path.dirname(os.path.abspath(__file__)),
              FLAGS.log_dir + FLAGS.model_name + '/model')


def train():
    training = tf.placeholder(tf.bool)
    global_step = tf.get_variable('global_step', trainable=False, shape=(),
                                  dtype=tf.int32,
                                  initializer=tf.zeros_initializer())
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs, y_values, y_shape)
    logits, ratio = model.inference(x, seq_length, training,FLAGS.sequence_len)
    ctc_loss = model.loss(logits, seq_length, y)
    opt = model.train_opt(FLAGS.step_rate,FLAGS.max_steps, global_step=global_step)
    step = opt.minimize(ctc_loss,global_step = global_step)
    error = model.prediction(logits, seq_length, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    save_model()
    if FLAGS.retrain == False:
        sess.run(init)
        print("Model init finished, begin loading data. \n")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(
            FLAGS.log_dir + FLAGS.model_name))
        print("Model loaded finished, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(
        FLAGS.log_dir + FLAGS.model_name + '/summary/', sess.graph)

    train_ds = read_tfrecord(FLAGS.data_dir, FLAGS.tfrecord, FLAGS.cache_file,
                             FLAGS.sequence_len, k_mer=FLAGS.k_mer,max_segments_num=FLAGS.segments_num)
    start = time.time()
    for i in range(FLAGS.max_steps):
        batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
        indxs, values, shape = batch_y
        feed_dict = {x: batch_x, seq_length: seq_len / ratio, y_indexs: indxs,
                     y_values: values, y_shape: shape,
                     training: True}
        loss_val, _ = sess.run([ctc_loss, step], feed_dict=feed_dict)
        if i % 10 == 0:
            global_step_val = tf.train.global_step(sess, global_step)
            valid_x, valid_len, valid_y = train_ds.next_batch(FLAGS.batch_size)
            indxs, values, shape = valid_y
            feed_dict = {x: valid_x, seq_length: valid_len / ratio,
                         y_indexs: indxs, y_values: values, y_shape: shape,
                         training: True}
            error_val = sess.run(error, feed_dict=feed_dict)
            end = time.time()
            print(
            "Step %d/%d Epoch %d, batch number %d, loss: %5.3f edit_distance: %5.3f Elapsed Time/step: %5.3f" \
            % (i, FLAGS.max_steps, train_ds.epochs_completed,
               train_ds.index_in_epoch, loss_val, error_val,
               (end - start) / (i + 1)))
            saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/model.ckpt',
                       global_step=global_step_val)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step_val)
            summary_writer.flush()
    global_step_val = tf.train.global_step(sess, global_step)
    print("Model %s saved." % (FLAGS.log_dir + FLAGS.model_name))
    print("Reads number %d" % (train_ds.reads_n))
    saver.save(sess, FLAGS.log_dir + FLAGS.model_name + '/final.ckpt',
               global_step=global_step_val)


def run(args):
    global FLAGS
    FLAGS = args
    FLAGS.data_dir = FLAGS.data_dir + os.path.sep
    FLAGS.log_dir = FLAGS.log_dir + os.path.sep
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training model with tfrecord file')
    parser.add_argument('-i', '--data_dir', required = True,
                        help="Directory that store the tfrecord files.")
    parser.add_argument('-o', '--log_dir', required = True  ,
                        help="log directory that store the training model.")
    parser.add_argument('-m', '--model_name', required = True,
                        help='model_name')
    parser.add_argument('-f', '--tfrecord', default="train.tfrecords",
                        help='tfrecord file')
    parser.add_argument('-c', '--cache_file', default=None, help="Cache file.")
    parser.add_argument('-s', '--sequence_len', type=int, default=400,
                        help='the length of sequence')
    parser.add_argument('-b', '--batch_size', type=int, default=200,
                        help='Batch size')
    parser.add_argument('-t', '--step_rate', type=float, default=1e-3,
                        help='Step rate')
    parser.add_argument('-x', '--max_steps', type=int, default=10000,
                        help='Maximum step')
    parser.add_argument('-n', '--segments_num', type = int, default = None,
                        help='Maximum number of segments read into the training queue, default(None) read all segments.')
    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')
    parser.add_argument('-r', '--retrain', type=bool, default=False,
                        help='flag if retrain or not')
    args = parser.parse_args(sys.argv[1:])
    if args.cache_file is None:
        args.cache_file = args.data_dir + '/cache.hdf5'
    run(args)
