# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Fri Aug 11 20:14:01 2017

import argparse

import numpy as np
import tensorflow as tf

from chiron.chiron_eval import inference
from chiron.chiron_input import read_data_for_eval


def input_output_list():
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    logits, _ = inference(x, seq_length, training=training)
    predict = tf.nn.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_length, merge_repeated=True)
    input_dict = {'x': x, 'seq_length': seq_length, 'training': training}
    output_dict = {'decoded_indices': predict[0][0].indices, 'decoded_values': predict[0][0].values,
                   'neg_sum_logits': predict[1]}
    return input_dict, output_dict


# if __name__=='__main__':
parser = argparse.ArgumentParser()
parser.add_argument("--export_folder", default="../model/crnn3+3_S10_2_re/export", type=str,
                    help="Frozen model file to import")
parser.add_argument('-m', '--model', default='../model/crnn3+3_S10_2_re', help="model folder")
parser.add_argument('-b', '--batch_size', type=int, default=1100,
                    help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
parser.add_argument('-l', '--segment_len', type=int, default=300, help="Segment length to be divided into.")
parser.add_argument('-s', '--start', type=int, default=0, help="Start index of the signal file.")
parser.add_argument('-j', '--jump', type=int, default=30, help="Step size for segment")
parser.add_argument('-t', '--threads', type=int, default=0, help="Threads number")
FLAGS = parser.parse_args()
with tf.Session() as sess:
    meta_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.export_folder)
    input_path = '/home/haotianteng/UQ/deepBNS/chiron_package/example_data/output/raw/IMB14_011406_LT_20170328_FNFAF13338_MN17027_mux_scan_S18_2_Watermanag_28032017_64349_ch7_read50_strand.signal'
    eval_data = read_data_for_eval(input_path, FLAGS.start, seg_length=FLAGS.segment_len, step=FLAGS.jump)
    batch_x, seq_len, _ = eval_data.next_batch(FLAGS.batch_size, shuffle=False)
    batch_x = np.pad(batch_x, ((0, FLAGS.batch_size - len(batch_x)), (0, 0)), mode='constant')
    seq_len = np.pad(seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='constant')

    # feed_dict = {input_dict['x']:batch_x,input_dict['seq_length']:seq_len,input_dict['training']:False}
    # sess.run(output_dict['decoded_indices'],feed_dict=feed_dict)
