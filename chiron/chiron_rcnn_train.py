#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Mon Mar 27 14:04:57 2017
# from rnn import rnn_layers
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import time
import argparse

import tensorflow as tf
import chiron.chiron_model as model
from chiron.chiron_input import read_tfrecord
from chiron.chiron_input import read_cache_dataset
from six.moves import range

DEFAULT_OFFSET = 10

def save_hyper_parameter():
    """
    TODO: Function to save the hyper parameter.
    """

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
    default_config = os.path.join(FLAGS.log_dir,FLAGS.model_name,'model.json')
    if FLAGS.retrain:
        if os.path.isfile(default_config):
            config_file = default_config
        else:
            raise ValueError("Model Json file has not been found in model log directory")
    else:
        config_file = FLAGS.configure   
    config = model.read_config(config_file)
    logits, ratio = model.inference(x, seq_length, training,FLAGS.sequence_len,configure = config)
    if 'fl_gamma' in config.keys():
        ctc_loss = model.loss(logits, seq_length, y, fl_gamma = config['fl_gamma'])
    else:
        ctc_loss = model.loss(logits, seq_length, y)
    opt = model.train_opt(FLAGS.step_rate,
                          FLAGS.max_steps, 
                          global_step=global_step,
                          opt_name = config['opt_method'])
    if FLAGS.gradient_clip is None:
        step = opt.minimize(ctc_loss,global_step = global_step)
    else:
        gradients, variables = zip(*opt.compute_gradients(ctc_loss))
        gradients = [None if gradient is None else tf.clip_by_norm(gradient, FLAGS.gradient_clip) for gradient in gradients]
        step = opt.apply_gradients(zip(gradients, variables),global_step = global_step)
    error = model.prediction(logits, seq_length, y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()
    print("Begin training using following setting:")
    for pro in dir(FLAGS):
        if not pro.startswith('_'):
            print("%s:%s"%(pro,getattr(FLAGS,pro)))
    sess = tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=FLAGS.threads,
                                            intra_op_parallelism_threads=FLAGS.threads,
                                            allow_soft_placement=True))
    if FLAGS.retrain == False:
        sess.run(init)
        print("Model init finished, begin loading data. \n")
    else:
        saver.restore(sess, tf.train.latest_checkpoint(
            FLAGS.log_dir + FLAGS.model_name))
        print("Model loaded finished, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(
        FLAGS.log_dir + FLAGS.model_name + '/summary/', sess.graph)
    model.save_model(default_config,config)
    train_ds,valid_ds = generate_train_valid_datasets(initial_offset = DEFAULT_OFFSET)
    start = time.time()
    resample_n = 0
    for i in range(FLAGS.max_steps):
        if FLAGS.resample_after_epoch == 0:
            pass
        elif train_ds.epochs_completed >= FLAGS.resample_after_epoch:
            train_ds,valid_ds = generate_train_valid_datasets(initial_offset = resample_n*FLAGS.offset_increment + DEFAULT_OFFSET)
        batch_x, seq_len, batch_y = train_ds.next_batch(FLAGS.batch_size)
        indxs, values, shape = batch_y
        feed_dict = {x: batch_x, seq_length: seq_len / ratio, y_indexs: indxs,
                     y_values: values, y_shape: shape,
                     training: True}
        loss_val, _ = sess.run([ctc_loss, step], feed_dict=feed_dict)
        if i % 10 == 0:
            global_step_val = tf.train.global_step(sess, global_step)
            valid_x, valid_len, valid_y = valid_ds.next_batch(FLAGS.batch_size)
            indxs, values, shape = valid_y
            feed_dict = {x: valid_x, seq_length: valid_len / ratio,
                         y_indexs: indxs, y_values: values, y_shape: shape,
                         training: True}
            error_val = sess.run(error, feed_dict=feed_dict)
            end = time.time()
            print(
            "Step %d/%d Epoch %d, batch number %d, train_loss: %5.3f validate_edit_distance: %5.3f Elapsed Time/step: %5.3f" \
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
    
def generate_train_valid_datasets(initial_offset = 10):
    if FLAGS.read_cache:
        train_ds = read_cache_dataset(FLAGS.train_cache)
        if FLAGS.validation is not None:
            valid_ds = read_cache_dataset(FLAGS.valid_cache)
        else:
            valid_ds = train_ds
        if train_ds.event.shape[1]!=FLAGS.sequence_len:
            raise ValueError("The event length of training cached dataset %d is inconsistent with given sequene_len %d"%(train_ds.event.shape()[1],FLAGS.sequence_len))
        if valid_ds.event.shape[1]!=FLAGS.sequence_len:
            raise ValueError("The event length of training cached dataset %d is inconsistent with given sequene_len %d"%(valid_ds.event.shape()[1],FLAGS.sequence_len))
        return train_ds,valid_ds
    sys.stdout.write("Begin reading training dataset.\n")
    train_ds = read_tfrecord(FLAGS.data_dir, 
                             FLAGS.tfrecord, 
                             FLAGS.train_cache,
                             FLAGS.sequence_len, 
                             k_mer=FLAGS.k_mer,
                             max_segments_num=FLAGS.segments_num,
                             skip_start = initial_offset)
    sys.stdout.write("Begin reading validation dataset.\n")
    if FLAGS.validation is not None:
        valid_ds = read_tfrecord(FLAGS.data_dir, 
                                 FLAGS.validation,
                                 FLAGS.valid_cache,
                                 FLAGS.sequence_len, 
                                 k_mer=FLAGS.k_mer,
                                 max_segments_num=FLAGS.segments_num)
    else:
        valid_ds = train_ds
    return train_ds,valid_ds
def run(args):
    global FLAGS
    FLAGS = args
    if FLAGS.train_cache is None:
        FLAGS.train_cache = FLAGS.data_dir + '/train_cache.hdf5'
    if (FLAGS.valid_cache is None) and (FLAGS.validation is not None):
        FLAGS.valid_cache = FLAGS.data_dir + '/valid_cache.hdf5'
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
    parser.add_argument('-v', '--validation', default = None, 
                        help="validation tfrecord file, default is None, which conduct no validation")
    parser.add_argument('-f', '--tfrecord', default="train.tfrecords",
                        help='tfrecord file')
    parser.add_argument('--train_cache', default=None, help="Cache file for training dataset.")
    parser.add_argument('--valid_cache', default=None, help="Cache file for validation dataset.")
    parser.add_argument('-s', '--sequence_len', type=int, default=400,
                        help='the length of sequence')
    parser.add_argument('-b', '--batch_size', type=int, default=300,
                        help='Batch size')
    parser.add_argument('-t', '--step_rate', type=float, default=4e-3,
                        help='Step rate')
    parser.add_argument('-x', '--max_steps', type=int, default=10000,
                        help='Maximum step')
    parser.add_argument('-n', '--segments_num', type = int, default = None,
                        help='Maximum number of segments read into the training queue, default(None) read all segments.')
    parser.add_argument('--configure', default = None,
                        help="Model structure configure json file.")
    parser.add_argument('-k', '--k_mer', default=1, help='Output k-mer size')
    parser.add_argument('--resample_after_epoch',
                        type = int,
                        default = 0, 
                        help='Resample the reads data every n epoches, with an increasing initial offset.')
    parser.add_argument('--threads',
                        type = int,
                        default = 0, 
                        help='The threads that available, if 0 use all threads that can be found.')
    parser.add_argument('--offset_increment',
                        type = int,
                        default = 3,
                        help='The increament of initial offset if the resample_after_epoch has been set.')
    parser.add_argument('--gradient_clip',
                        type = float,
                        default = None,
                        help = 'Clip the gradient by the gradient_clip x normalization, a good estimate is 5.')
    parser.add_argument('--retrain', dest='retrain', action='store_true',
                        help='Set retrain to true')
    parser.add_argument('--read_cache',dest='read_cache',action='store_true',
                        help="Read from cached hdf5 file.")
    parser.set_defaults(retrain=False)
    args = parser.parse_args(sys.argv[1:])
    run(args)
