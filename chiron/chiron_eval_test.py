# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Sun Apr 30 11:59:15 2017
from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from chiron import chiron_model
from chiron.chiron_input import read_data_for_eval
from chiron.cnn import getcnnfeature
from chiron.cnn import getcnnlogit
from chiron.rnn import rnn_layers
from chiron.utils.easy_assembler import simple_assembly
from chiron.utils.easy_assembler import simple_assembly_qs
from chiron.utils.easy_assembler import global_alignment_assembly
from chiron.utils.unix_time import unix_time
from chiron.utils.progress import multi_pbars
from six.moves import range
import threading
from collections import defaultdict

def sparse2dense(predict_val):
    """Transfer a sparse input in to dense representation
    Args:
        predict_val ((docded, log_probabilities)): Tuple of shape 2, output from the tf.nn.ctc_beam_search_decoder or tf.nn.ctc_greedy_decoder.
            decoded:A list of length `top_paths`, where decoded[j] is a SparseTensor containing the decoded outputs:
                decoded[j].indices: Matrix of shape [total_decoded_outputs[j], 2], each row stand for [batch, time] index in dense representation.
                decoded[j].values: Vector of shape [total_decoded_outputs[j]]. The vector stores the decoded classes for beam j.
                decoded[j].shape: Vector of shape [2]. Give the [batch_size, max_decoded_length[j]].
            Check the format of the sparse tensor at https://www.tensorflow.org/api_docs/python/tf/SparseTensor
            log_probability: A float matrix of shape [batch_size, top_paths]. Give the sequence log-probabilities.

    Returns:
        predict_read[Float]: Nested List, [path_index][read_index][base_index], give the list of decoded reads(in number representation 0-A, 1-C, 2-G, 3-T).
        uniq_list[Int]: Nested List, [top_paths][batch_index], give the batch index that exist in the decoded output.
    """

    predict_val_top5 = predict_val[0]
    predict_read = list()
    uniq_list = list()
    for i in range(len(predict_val_top5)):
        predict_val = predict_val_top5[i]
        unique, pre_counts = np.unique(
            predict_val.indices[:, 0], return_counts=True)
        uniq_list.append(unique)
        pos_predict = 0
        predict_read_temp = list()
        for indx, _ in enumerate(pre_counts):
            predict_read_temp.append(
                predict_val.values[pos_predict:pos_predict + pre_counts[indx]])
            pos_predict += pre_counts[indx]
        predict_read.append(predict_read_temp)
    return predict_read, uniq_list


def index2base(read):
    """Transfer the number into dna base.
    The transfer will go through each element of the input int vector.
    Args:
        read (Int): An Iterable item containing element of [0,1,2,3].

    Returns:
        bpread (Char): A String containing translated dna base sequence.
    """

    base = ['A', 'C', 'G', 'T']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


def path_prob(logits):
    """Calculate the mean of the difference between highest and second highest logits in path.
    Given the p_i = exp(logit_i)/sum_k(logit_k), we can get the quality score for the concensus sequence as:
        qs = 10 * log_10(p1/p2) = 10 * log_10(exp(logit_1 - logit_2)) = 10 * ln(10) * (logit_1 - logit_2), 
        where p_1,logit_1 are the highest probability, logit, and p_2, logit_2 are the second highest probability logit.

    Args:
        logits (Float): Tensor of shape [batch_size, max_time,class_num], output logits.

    Returns:
        prob_logits(Float): Tensor of shape[batch_size].
    """

    fea_shape = tf.shape(logits)
    bsize = fea_shape[0]
    seg_len = fea_shape[1]
    top2_logits = tf.nn.top_k(logits, k=2)[0]
    logits_diff = tf.slice(top2_logits, [0, 0, 0], [bsize, seg_len, 1]) - tf.slice(
        top2_logits, [0, 0, 1], [bsize, seg_len, 1])
    prob_logits = tf.reduce_mean(logits_diff, axis=-2)
    return prob_logits


def qs(consensus, consensus_qs, output_standard='phred+33'):
    """Calculate the quality score for the consensus read.

    Args:
        consensus (Int): 2D Matrix (read length, bases) given the count of base on each position.
        consensus_qs (Float): 1D Vector given the mean of the difference between the highest logit and second highest logit.
        output_standard (str, optional): Defaults to 'phred+33'. Quality score output format.

    Returns:
        quality score: Return the queality score as int or string depending on the format.
    """

    sort_ind = np.argsort(consensus, axis=0)
    L = consensus.shape[1]
    sorted_consensus = consensus[sort_ind, np.arange(L)[np.newaxis, :]]
    sorted_consensus_qs = consensus_qs[sort_ind, np.arange(L)[np.newaxis, :]]
    quality_score = 10 * (np.log10((sorted_consensus[3, :] + 1) / (
        sorted_consensus[2, :] + 1))) + sorted_consensus_qs[3, :] / sorted_consensus[3, :] / np.log(10)
    if output_standard == 'number':
        return quality_score.astype(int)
    elif output_standard == 'phred+33':
        q_string = [chr(x + 33) for x in quality_score.astype(int)]
        return ''.join(q_string)



#def run(args):
#    global FLAGS
#    FLAGS = args
#    # logging.debug("Flags:\n%s", pformat(vars(args)))
#    time_dict = unix_time(evaluation)
#    print(FLAGS.output)
#    print('Real time:%5.3f Systime:%5.3f Usertime:%5.3f' %
#          (time_dict['real'], time_dict['sys'], time_dict['user']))
#    meta_folder = os.path.join(FLAGS.output, 'meta')
#    if os.path.isdir(FLAGS.input):
#        file_pre = 'all'
#    else:
#        file_pre = os.path.splitext(os.path.basename(FLAGS.input))[0]
#    path_meta = os.path.join(meta_folder, file_pre + '.meta')
#    with open(path_meta, 'a+') as out_meta:
#        out_meta.write("# Wall_time Sys_time User_time Cpu_time\n")
#        out_meta.write("%5.3f %5.3f %5.3f %5.3f\n" % (
#            time_dict['real'], time_dict['sys'], time_dict['user'], time_dict['sys'] + time_dict['user']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='chiron',
                                     description='A deep neural network basecaller.')
    parser.add_argument('-i', '--input', required = True,
                        help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output Folder name")
    parser.add_argument('-m', '--model', required = True,
                        help="model folder path")
    parser.add_argument('-s', '--start', type=int, default=0,
                        help="Start index of the signal file.")
    parser.add_argument('-b', '--batch_size', type=int, default=400,
                        help="Batch size for run, bigger batch_size will increase the processing speed and give a slightly better accuracy but require larger RAM load")
    parser.add_argument('-l', '--segment_len', type=int, default=400,
                        help="Segment length to be divided into.")
    parser.add_argument('-j', '--jump', type=int, default=30,
                        help="Step size for segment")
    parser.add_argument('-t', '--threads', type=int, default=0,
                        help="Threads number")
    parser.add_argument('-e', '--extension', default='fastq',
                        help="Output file extension.")
    parser.add_argument('--beam', type=int, default=30,
                        help="Beam width used in beam search decoder, default is 0, in which a greedy decoder is used. Recommend width:100, Large beam width give better decoding result but require longer decoding time.")
    parser.add_argument('--concise', action='store_true',
                        help="Concisely output the result, the meta and segments files will not be output.")
    parser.add_argument('--mode', default = 'dna',
                        help="Output mode, can be chosen from dna or rna.")
    FLAGS = parser.parse_args(sys.argv[1:])

    pbars = multi_pbars(["Logits(batches)","ctc(batches)","logits(files)","ctc(files)"])
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    config_path = os.path.join(FLAGS.model,'model.json')
    model_configure = chiron_model.read_config(config_path)
    logits, ratio = chiron_model.inference(
                                    x, 
                                    seq_length, 
                                    training=training,
                                    full_sequence_len = FLAGS.segment_len,
                                    configure = model_configure)
    prorbs=tf.nn.softmax(logits)
    predict = tf.nn.ctc_beam_search_decoder(
            tf.transpose(logits, perm=[1, 0, 2]),
            seq_length, merge_repeated=False,
            beam_width=FLAGS.beam)
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=FLAGS.threads,
                            inter_op_parallelism_threads=FLAGS.threads)
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver(var_list=tf.trainable_variables()+tf.moving_average_variables())
    sess = tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=config))
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model))
    if os.path.isdir(FLAGS.input):
        file_list = os.listdir(FLAGS.input)
        file_dir = FLAGS.input
    else:
        file_list = [os.path.basename(FLAGS.input)]
        file_dir = os.path.abspath(
            os.path.join(FLAGS.input, os.path.pardir))
    for file in file_list:
        file_path = os.path.join(file_dir,file)
        eval_data = read_data_for_eval(file_path, 
                                       FLAGS.start,
                                       seg_length=FLAGS.segment_len,
                                       step=FLAGS.jump)
        reads_n = eval_data.reads_n
        for i in range(0, reads_n, FLAGS.batch_size):
            batch_x, seq_len, _ = eval_data.next_batch(
                FLAGS.batch_size, shuffle=False)
            batch_x = np.pad(
                batch_x, ((0, FLAGS.batch_size - len(batch_x)), (0, 0)), mode='wrap')
            seq_len = np.pad(
                seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='wrap')
            batch_x[0,:] = batch_x[0,:]
            feed_dict = {
                x: batch_x,
                seq_length: np.round(seq_len/ratio).astype(np.int32),
                training: True,
            }
            prob_val, logits_val,predict_val = sess.run([prorbs,logits,predict], feed_dict = feed_dict)
    predict_read, unique = sparse2dense(predict_val)
    sig_index=0
    print(predict_read[0][0])
    def plot_signal(INDEX):
        COLOR=['red','yellow','green','blue','--']
        plt.rcParams["figure.figsize"] = [15,15]
        plt1 = plt.subplot(211)
        plt2 = plt.subplot(212)
        x1=range(2000)
        x2=range(0,2000,5)
        plt1.plot(x1,batch_x[INDEX,:],'black')
        for i in range(5):
            plt2.plot(x2,prob_val[INDEX,:,i],COLOR[i])        
    plot_signal(sig_index)