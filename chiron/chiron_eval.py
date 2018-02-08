#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:59:15 2017

@author: haotianteng
"""
import argparse
import os
import time
import sys
import numpy as np
import tensorflow as tf
from chiron.chiron_input import read_data_for_eval
from chiron.utils.easy_assembler import simple_assembly, simple_assembly_qs
from chiron.cnn import getcnnfeature
from chiron.rnn import rnn_layers
from chiron.utils.unix_time import unix_time


def inference(x, seq_length, training):
    cnn_feature = getcnnfeature(x, training=training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = int(FLAGS.segment_len / feashape[1])
    logits = rnn_layers(cnn_feature, seq_length // ratio, training, class_n=5)
    return logits, ratio


def sparse2dense(predict_val):
    predict_val_top5 = predict_val[0]
    predict_read = list()
    uniq_list = list()
    for i in range(len(predict_val_top5)):
        predict_val = predict_val_top5[i]
        unique, pre_counts = np.unique(predict_val.indices[:, 0], return_counts=True)
        uniq_list.append(unique)
        pos_predict = 0
        predict_read_temp = list()
        for indx, counts in enumerate(pre_counts):
            predict_read_temp.append(predict_val.values[pos_predict:pos_predict + pre_counts[indx]])
            pos_predict += pre_counts[indx]
        predict_read.append(predict_read_temp)
    return predict_read, uniq_list


def index2base(read):
    base = ['A', 'C', 'G', 'T']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


def path_prob(logits):
    top2_logits = tf.nn.top_k(logits, k=2)[0]
    logits_diff = tf.slice(top2_logits,
                           [0, 0, 0],
                           [FLAGS.batch_size,
                            FLAGS.segment_len, 1]) - \
                  tf.slice(top2_logits,
                           [0, 0, 1],
                           [FLAGS.batch_size,
                            FLAGS.segment_len, 1])
    prob_logits = tf.reduce_mean(logits_diff, axis=-2)
    return prob_logits


def qs(consensus, consensus_qs, output_standard='phred+33'):
    sort_ind = np.argsort(consensus, axis=0)
    l = consensus.shape[1]
    sorted_consensus = consensus[sort_ind, np.arange(l)[np.newaxis, :]]
    sorted_consensus_qs = consensus_qs[sort_ind, np.arange(l)[np.newaxis, :]]
    quality_score = 10 * (np.log10((sorted_consensus[3, :] + 1) / (sorted_consensus[2, :] + 1))) + \
                    sorted_consensus_qs[3, :] / sorted_consensus[3, :] / np.log(10)
    if output_standard == 'number':
        return quality_score.astype(int)
    elif output_standard == 'phred+33':
        q_string = [chr(x + 33) for x in quality_score.astype(int)]
        return ''.join(q_string)


def write_output(segments, consensus, time_list, file_pre, concise=False, suffix='fasta', seg_q_score=None,
                 q_score=None):
    """
    seg_q_score: A length seg_num string list. Quality score for the segments.
    q_socre: A string. Quality score for the consensus sequence.
    """
    start_time, reading_time, basecall_time, assembly_time = time_list
    result_folder = os.path.join(FLAGS.output, 'result')
    seg_folder = os.path.join(FLAGS.output, 'segments')
    meta_folder = os.path.join(FLAGS.output, 'meta')
    path_con = os.path.join(result_folder, file_pre + '.' + suffix)
    if not concise:
        path_reads = os.path.join(seg_folder, file_pre + '.' + suffix)
        path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_reads, 'w+') as out_f, open(path_con, 'w+') as out_con:
        if not concise:
            for indx, read in enumerate(segments):
                out_f.write(file_pre + str(indx) + '\n')
                out_f.write(read + '\n')
                if (suffix == 'fastq') and (seg_q_score is not None):
                    out_f.write('+\n')
                    out_f.write(seg_q_score[indx] + '\n')
        if (suffix == 'fastq') and (q_score is not None):
            out_con.write('@{}\n{}\n+\n{}\n'.format(file_pre, consensus, q_score))
        else:
            out_con.write('{}\n{}'.format(file_pre, consensus))
    if not concise:
        with open(path_meta, 'w+') as out_meta:
            total_time = time.time() - start_time
            output_time = total_time - assembly_time
            assembly_time -= basecall_time
            basecall_time -= reading_time
            total_len = len(consensus)
            total_time = time.time() - start_time
            out_meta.write("# Reading Basecalling assembly output total rate(bp/s)\n")
            out_meta.write("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n" % (
                reading_time, basecall_time, assembly_time, output_time, total_time, total_len / total_time))
            out_meta.write("# read_len batch_size segment_len jump start_pos\n")
            out_meta.write(
                "%d %d %d %d %d\n" % (total_len, FLAGS.batch_size, FLAGS.segment_len, FLAGS.jump, FLAGS.start))
            out_meta.write("# input_name model_name\n")
            out_meta.write("%s %s\n" % (FLAGS.input, FLAGS.model))


def evaluation():
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    logits, _ = inference(x, seq_length, training=training)
    if FLAGS.extension == 'fastq':
        prob = path_prob(logits)
    if FLAGS.beam == 0:
        predict = tf.nn.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_length, merge_repeated=True)
    else:
        predict = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_length, merge_repeated=False,
                                                beam_width=FLAGS.beam)  # For beam_search_decoder, set the merge_repeated to false. 5-10 times slower than greedy decoder
    config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=FLAGS.threads,
                            inter_op_parallelism_threads=FLAGS.threads)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.model))
        if os.path.isdir(FLAGS.input):
            file_list = os.listdir(FLAGS.input)
            file_dir = FLAGS.input
        else:
            file_list = [os.path.basename(FLAGS.input)]
            file_dir = os.path.abspath(os.path.join(FLAGS.input, os.path.pardir))
        # Make output folder.
        if not os.path.exists(FLAGS.output):
            os.makedirs(FLAGS.output)
        if not os.path.exists(os.path.join(FLAGS.output, 'segments')):
            os.makedirs(os.path.join(FLAGS.output, 'segments'))
        if not os.path.exists(os.path.join(FLAGS.output, 'result')):
            os.makedirs(os.path.join(FLAGS.output, 'result'))
        if not os.path.exists(os.path.join(FLAGS.output, 'meta')):
            os.makedirs(os.path.join(FLAGS.output, 'meta'))
        ##
        for name in file_list:
            start_time = time.time()
            if not name.endswith('.signal'):
                continue
            file_pre = os.path.splitext(name)[0]
            input_path = os.path.join(file_dir, name)
            eval_data = read_data_for_eval(input_path, FLAGS.start, seg_length=FLAGS.segment_len, step=FLAGS.jump,
                                           sig_norm=False)
            reads_n = eval_data.reads_n
            reading_time = time.time() - start_time
            reads = list()
            signals = np.empty((0, FLAGS.segment_len), dtype=np.float)
            qs_list = np.empty((0, 1), dtype=np.float)
            qs_string = None
            for i in range(0, reads_n, FLAGS.batch_size):
                batch_x, seq_len, _ = eval_data.next_batch(FLAGS.batch_size, shuffle=False, sig_norm=False)
                if not FLAGS.concise:
                    signals += batch_x
                batch_x = np.pad(batch_x, ((0, FLAGS.batch_size - len(batch_x)), (0, 0)), mode='constant')
                seq_len = np.pad(seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='constant')
                feed_dict = {x: batch_x, seq_length: seq_len, training: False}
                if FLAGS.extension == 'fastq':
                    predict_val, logits_prob = sess.run([predict, prob], feed_dict=feed_dict)
                else:
                    predict_val = sess.run(predict, feed_dict=feed_dict)
                predict_read, unique = sparse2dense(predict_val)
                predict_read = predict_read[0]
                unique = unique[0]

                if FLAGS.extension == 'fastq':
                    logits_prob = logits_prob[unique]
                if i + FLAGS.batch_size > reads_n:
                    predict_read = predict_read[:reads_n - i]
                    if FLAGS.extension == 'fastq':
                        logits_prob = logits_prob[:reads_n - i]
                if FLAGS.extension == 'fastq':
                    qs_list = np.concatenate((qs_list, logits_prob))
                reads += predict_read
            print("Segment reads base calling finished, begin to assembly. %5.2f seconds" % (time.time() - start_time))
            basecall_time = time.time() - start_time
            bpreads = [index2base(read) for read in reads]
            if FLAGS.extension == 'fastq':
                consensus, qs_consensus = simple_assembly_qs(bpreads, qs_list)
                qs_string = qs(consensus, qs_consensus)
            else:
                consensus = simple_assembly(bpreads)
            if signals != eval_data.event:
                print(len(signals))
                print(signals)
                print(len(eval_data.event))
                print(eval_data.event)
            c_bpread = index2base(np.argmax(consensus, axis=0))
            np.set_printoptions(threshold=np.nan)
            assembly_time = time.time() - start_time
            print("Assembly finished, begin output. %5.2f seconds" % (time.time() - start_time))
            list_of_time = [start_time, reading_time, basecall_time, assembly_time]
            write_output(bpreads, c_bpread, list_of_time, file_pre, concise=FLAGS.concise, suffix=FLAGS.extension,
                         q_score=qs_string)


def run(args):
    global FLAGS
    FLAGS = args
    time_dict = unix_time(evaluation)
    print(FLAGS.output)
    print('Real time:%5.3f Systime:%5.3f Usertime:%5.3f' % (time_dict['real'], time_dict['sys'], time_dict['user']))
    meta_folder = os.path.join(FLAGS.output, 'meta')
    if os.path.isdir(FLAGS.input):
        file_pre = 'all'
    else:
        file_pre = os.path.splitext(os.path.basename(FLAGS.input))[0]
    path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_meta, 'a+') as out_meta:
        out_meta.write("# Wall_time Sys_time User_time Cpu_time\n")
        out_meta.write("%5.3f %5.3f %5.3f %5.3f\n" % (
            time_dict['real'], time_dict['sys'], time_dict['user'], time_dict['sys'] + time_dict['user']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='chiron', description='A deep neural network basecaller.')
    parser.add_argument('-i', '--input', default='example_data/output/raw',
                        help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o', '--output', default='example_data/output', help="Output Folder name")
    parser.add_argument('-m', '--model', default='model/DNA_default', help="model folder")
    parser.add_argument('-s', '--start', type=int, default=0, help="Start index of the signal file.")
    parser.add_argument('-b', '--batch_size', type=int, default=1100,
                        help="Batch size for run, bigger batch_size will increase the processing speed and give a slightly better accuracy but require larger RAM load")
    parser.add_argument('-l', '--segment_len', type=int, default=300, help="Segment length to be divided into.")
    parser.add_argument('-j', '--jump', type=int, default=30, help="Step size for segment")
    parser.add_argument('-t', '--threads', type=int, default=0, help="Threads number")
    parser.add_argument('-e', '--extension', default='fastq', help="Output file extension.")
    parser.add_argument('--beam', type=int, default=0,
                        help="Beam width used in beam search decoder, default is 0, in which a greedy decoder is used. Recommend width:100, Large beam width give better decoding result but require longer decoding time.")
    parser.add_argument('--concise', action='store_true',
                        help="Concisely output the result, the meta and segments files will not be output.")
    args = parser.parse_args(sys.argv[1:])
    run(args)
