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
from collections import namedtuple

SparseTensor = namedtuple("SparseTensor","indices values dense_shape")
tf.logging.set_verbosity(tf.logging.ERROR)
def sparse2dense(predict_val):
    """Transfer a sparse input in to dense representation
    Args:
        predict_val (docoded, log_probabilities): Tuple of shape 2, output from the tf.nn.ctc_beam_search_decoder or tf.nn.ctc_greedy_decoder.
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
    for decode in predict_val_top5:
        unique, pre_counts = np.unique(
            decode.indices[:, 0], return_counts=True)
        uniq_list.append(unique)
        pos_predict = 0
        predict_read_temp = list()
        for indx, _ in enumerate(pre_counts):
            predict_read_temp.append(
                decode.values[pos_predict:pos_predict + pre_counts[indx]])
            pos_predict += pre_counts[indx]
        predict_read.append(predict_read_temp)
    return predict_read, uniq_list

def slice_sparse_tensor(input_sp,start,end):
    """
    Slice the given sparse_tensor from start to end by batches.
    Args:
        input_sp: the input sparse tensor
        start: start index.
        end: end index, the boundary behaviour is like numpy indexing. e.g. start<=array<end
    Return:
        sliced_sp: The slicing sparse tensor.
    """
    axis = 0
    mask = np.logical_and(input_sp.indices[:,axis]>=start,input_sp.indices[:,axis]<end)
    new_indices=input_sp.indices[mask] - [start,0]
    return SparseTensor(indices=new_indices,
                        values=input_sp.values[mask],
                        dense_shape=np.asarray([end-start,input_sp.dense_shape[1-axis]]))
    
def slice_ctc_decoding_result(input_decode,start,end):
    """
    Slice the result from tf.nn.ctc_beamsearch_decoder
    Args:
        input_decode: The result from the beam search decoder.
        start: slicing batch start index.
        end: slicing batch end index.
    """
    decodes = input_decode[0]
    log_p = input_decode[1]
    slices = list()
    for decode in decodes:
        slices.append(slice_sparse_tensor(decode,start,end))
    return (slices,log_p[start:end,:])

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

def get_assembler_kernal(jump, segment_len):
    """
    Args:
        jump: jump size
        segment_len: length of segment
    """
    #assembler='global'
    assembler='simple'
    if jump > 0.9*segment_len:
        assembler='glue'
    if jump >= segment_len:
        assembler='stick'
    return assembler

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

def write_output(segments, 
                 consensus, 
                 time_list, 
                 file_pre,
                 global_setting,
                 concise=False, 
                 suffix='fasta', 
                 seg_q_score=None,
                 q_score=None
                 ):
    """Write the output to the fasta(q) file.

    Args:
        segments ([Int]): List of read integer segments.
        consensus (str): String of the read represented in AGCT.
        time_list (Tuple): Tuple of time records.
        file_pre (str): Output fasta(q) file name(prefix).
        concise (bool, optional): Defaults to False. If False, the time records and segments will not be output.
        suffix (str, optional): Defaults to 'fasta'. Output file suffix from 'fasta', 'fastq'.
        seg_q_score ([str], optional): Defaults to None. Quality scores of read segment.
        q_score (str, optional): Defaults to None. Quality scores of the read.
        global_setting: The global Flags of chiron_eval.
    """
    start_time, reading_time, basecall_time, assembly_time = time_list
    result_folder = os.path.join(global_setting.output, 'result')
    seg_folder = os.path.join(global_setting.output, 'segments')
    meta_folder = os.path.join(global_setting.output, 'meta')
    path_con = os.path.join(result_folder, file_pre + '.' + suffix)
    if global_setting.mode == 'rna':
        consensus = consensus.replace('T','U').replace('t','u')
    if not concise:
        path_reads = os.path.join(seg_folder, file_pre + '.' + suffix)
        path_meta = os.path.join(meta_folder, file_pre + '.meta')
    with open(path_con, 'w+') as out_con:
        if not concise:
            with open(path_reads, 'w+') as out_f:
                for indx, read in enumerate(segments):
                    out_f.write('>{}{}\n{}\n'.format(file_pre, str(indx),read))
                    if (suffix == 'fastq') and (seg_q_score is not None):
                        out_f.write('@{}{}\n{}\n+\n{}\n'.format(file_pre, str(indx),read,seg_q_score[indx]))
        if (suffix == 'fastq') and (q_score is not None):
            out_con.write(
                '@{}\n{}\n+\n{}\n'.format(file_pre, consensus, q_score))
        else:
            out_con.write('>{}\n{}'.format(file_pre, consensus))
    if not concise:
        with open(path_meta, 'w+') as out_meta:
            total_time = time.time() - start_time
            output_time = total_time - assembly_time
            assembly_time -= basecall_time
            basecall_time -= reading_time
            total_len = len(consensus)
            total_time = time.time() - start_time
            out_meta.write(
                "# Reading Basecalling assembly output total rate(bp/s)\n")
            out_meta.write("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n" % (
                reading_time, basecall_time, assembly_time, output_time, total_time, total_len / total_time))
            out_meta.write(
                "# read_len batch_size segment_len jump start_pos\n")
            out_meta.write(
                "%d %d %d %d %d\n" % (total_len, 
                                      global_setting.batch_size, 
                                      global_setting.segment_len, 
                                      global_setting.jump, 
                                      global_setting.start))
            out_meta.write("# input_name model_name\n")
            out_meta.write("%s %s\n" % (global_setting.input, global_setting.model))
            
def compile_eval_graph(model_configure):
    class net:
        def __init__(self,configure):
            self.pbars = multi_pbars(["Logits(batches)","ctc(batches)","logits(files)","ctc(files)"])
            self.x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
            self.seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
            self.training = tf.placeholder(tf.bool)
            self.logits, self.ratio = chiron_model.inference(
                                            self.x, 
                                            self.seq_length, 
                                            training=self.training,
                                            full_sequence_len = FLAGS.segment_len,
                                            configure = configure)
            self.config = tf.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=FLAGS.threads,
                                    inter_op_parallelism_threads=FLAGS.threads)
            self.config.gpu_options.allow_growth = True
            self.logits_index = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
            self.logits_fname = tf.placeholder(tf.string, shape=[FLAGS.batch_size])
            self.logits_queue = tf.FIFOQueue(
                capacity=1000,
                dtypes=[tf.float32, tf.string, tf.int32, tf.int32],
                shapes=[self.logits.shape,self.logits_fname.shape,self.logits_index.shape, self.seq_length.shape]
            )
            self.logits_queue_size = self.logits_queue.size()
            self.logits_enqueue = self.logits_queue.enqueue((self.logits, self.logits_fname, self.logits_index, self.seq_length))
            self.logits_queue_close = self.logits_queue.close()
            ### Decoding logits into bases
            self.decode_predict_op, self.decode_prob_op, self.decoded_fname_op, self.decode_idx_op, self.decode_queue_size = decoding_queue(self.logits_queue)
            self.saver = tf.train.Saver(var_list=tf.trainable_variables()+tf.moving_average_variables())
        
        def init_session(self):
            self.sess = tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=self.config))
            self.saver.restore(self.sess, tf.train.latest_checkpoint(FLAGS.model))
            if os.path.isdir(FLAGS.input):
                self.file_list = os.listdir(FLAGS.input)
                self.file_dir = FLAGS.input
            else:
                self.file_list = [os.path.basename(FLAGS.input)]
                self.file_dir = os.path.abspath(
                    os.path.join(FLAGS.input, os.path.pardir))
            file_n = len(self.file_list)
            self.pbars.update(2,total = file_n)
            self.pbars.update(3,total = file_n)
            if not os.path.exists(FLAGS.output):
                os.makedirs(FLAGS.output)
            if not os.path.exists(os.path.join(FLAGS.output, 'segments')):
                os.makedirs(os.path.join(FLAGS.output, 'segments'))
            if not os.path.exists(os.path.join(FLAGS.output, 'result')):
                os.makedirs(os.path.join(FLAGS.output, 'result'))
            if not os.path.exists(os.path.join(FLAGS.output, 'meta')):
                os.makedirs(os.path.join(FLAGS.output, 'meta'))

        def _worker_fn(self):
            batch_x = np.asarray([[]]).reshape(0,FLAGS.segment_len)
            seq_len = np.asarray([])
            logits_idx = np.asarray([])
            logits_fn = np.asarray([])
            for f_i, name in enumerate(self.file_list):
                if not name.endswith('.signal'):
                    continue
                input_path = os.path.join(self.file_dir, name)
                eval_data = read_data_for_eval(input_path, FLAGS.start,
                                               seg_length=FLAGS.segment_len,
                                               step=FLAGS.jump)
                reads_n = eval_data.reads_n
                self.pbars.update(0,total = reads_n,progress = 0)
                self.pbars.update_bar()
                i=0
                while(eval_data.epochs_completed == 0):
                    current_batch, current_seq_len, _ = eval_data.next_batch(
                        FLAGS.batch_size-len(batch_x), shuffle=False)
                    current_n = len(current_batch)
                    batch_x = np.concatenate((batch_x,current_batch),axis = 0)
                    seq_len = np.concatenate((seq_len,current_seq_len),axis = 0)
                    logits_idx = np.concatenate((logits_idx,[i]*current_n),axis = 0)
                    logits_fn = np.concatenate((logits_fn,[name]*current_n),axis = 0)
                    i+=current_n
                    if len(batch_x) < FLAGS.batch_size:
                        self.pbars.update(0,progress=i)
                        self.pbars.update_bar()
                        continue
                    feed_dict = {
                        self.x.name: batch_x,
                        self.seq_length.name: np.round(seq_len/self.ratio).astype(np.int32),
                        self.training.name: False,
                        self.logits_index.name:logits_idx,
                        self.logits_fname.name:logits_fn,
                    }
                    #Training: Set it to  True for a temporary fix of the batch normalization problem: https://github.com/haotianteng/Chiron/commit/8fce3a3b4dac8e9027396bb8c9152b7b5af953ce
                    #TODO: change the training FLAG back to False after the new model has been trained.
                    self.sess.run(self.logits_enqueue,feed_dict=feed_dict)
                    batch_x = np.asarray([[]]).reshape(0,FLAGS.segment_len)
                    seq_len = np.asarray([])
                    logits_idx = np.asarray([])
                    logits_fn = np.asarray([])
                    self.pbars.update(0,progress=i)
                    self.pbars.update_bar()
                self.pbars.update(2,progress = f_i+1)
                self.pbars.update_bar()
            ### All files has been processed.
            batch_n = len(batch_x)
            if batch_n >0:
                pad_width = FLAGS.batch_size - batch_n
                batch_x = np.pad(
                        batch_x, ((0, pad_width), (0, 0)), mode='wrap')
                seq_len = np.pad(
                        seq_len, ((0, pad_width)), mode='wrap')
                logits_idx = np.pad(logits_idx,(0,pad_width),mode = 'constant',constant_values=-1)
                logits_fn = np.pad(logits_fn,(0,pad_width),mode = 'constant',constant_values='')
                self.sess.run(self.logits_enqueue,feed_dict = {
                        self.x.name: batch_x,
                        self.seq_length.name: np.round(seq_len/self.ratio).astype(np.int32),
                        self.training.name: False,
                        self.logits_index.name:logits_idx,
                        self.logits_fname.name:logits_fn,
                    })
            self.sess.run(self.logits_queue_close)
        def run_worker(self):
            worker = threading.Thread(target=self._worker_fn)
            worker.setDaemon(True)
            worker.start()
    eval_net = net(model_configure)
    eval_net.init_session()
    eval_net.run_worker()
    return eval_net
    

def evaluation():
    config_path = os.path.join(FLAGS.model,'model.json')
    model_configure = chiron_model.read_config(config_path)
    net = compile_eval_graph(model_configure)
    val = defaultdict(dict)  # We could read vals out of order, that's why it's a dict
    for f_i, name in enumerate(net.file_list):
        start_time = time.time()
        if not name.endswith('.signal'):
            continue
        file_pre = os.path.splitext(name)[0]
        input_path = os.path.join(net.file_dir, name)
        if FLAGS.mode == 'rna':
            eval_data = read_data_for_eval(input_path, FLAGS.start,
                                       seg_length=FLAGS.segment_len,
                                       step=FLAGS.jump)
        else:
            eval_data = read_data_for_eval(input_path, FLAGS.start,
                                       seg_length=FLAGS.segment_len,
                                       step=FLAGS.jump)
        reads_n = eval_data.reads_n
        net.pbars.update(1,total = reads_n,progress = 0)
        net.pbars.update_bar()
        reading_time = time.time() - start_time
        reads = list()
        if 'total_count' not in val[name].keys():
            val[name]['total_count'] = 0
        if 'index_list' not in val[name].keys():
            val[name]['index_list'] = []
        while True:
            l_sz, d_sz = net.sess.run([net.logits_queue_size, net.decode_queue_size])   
            if val[name]['total_count'] == reads_n:
                net.pbars.update(1,progress = val[name]['total_count'])
                break
            decode_ops = [net.decoded_fname_op, net.decode_idx_op, net.decode_predict_op, net.decode_prob_op]
            decoded_fname, i, predict_val, logits_prob = net.sess.run(decode_ops, feed_dict={net.training: False})
            decoded_fname = np.asarray([x.decode("UTF-8") for x in decoded_fname])
            ##Have difficulties integrate it into the tensorflow graph, as the number of file names in a batch is variable.
            ##And for loop can't be implemented as the eager execution is disabled due to the use of queue.
            uniq_fname,uniq_fn_idx = np.unique(decoded_fname,return_index = True)
            for fn_idx,fn in enumerate(uniq_fname):
                i = uniq_fn_idx[fn_idx]
                if fn != '':
                    occurance = np.where(decoded_fname==fn)[0]
                    start = occurance[0]
                    end = occurance[-1]+1
                    assert(len(occurance)==end-start)
                    if 'total_count' not in val[fn].keys():
                        val[fn]['total_count'] = 0
                    if 'index_list' not in val[fn].keys():
                        val[fn]['index_list'] = []
                    val[fn]['total_count'] += (end-start)
                    val[fn]['index_list'].append(i)
                    sliced_sparse = slice_ctc_decoding_result(predict_val,start,end)                
                    val[fn][i] = (sliced_sparse, logits_prob[decoded_fname==fn])               
            net.pbars.update(1,progress = val[name]['total_count'])
            net.pbars.update_bar()

        net.pbars.update(3,progress = f_i+1)
        net.pbars.update_bar()
        qs_list = np.empty((0, 1), dtype=np.float)
        qs_string = None
        for i in np.sort(val[name]['index_list']):
            predict_val, logits_prob = val[name][i]
            predict_read, unique = sparse2dense(predict_val)
            predict_read = predict_read[0]
            unique = unique[0]

            if FLAGS.extension == 'fastq':
                logits_prob = logits_prob[unique]
            if FLAGS.extension == 'fastq':
                qs_list = np.concatenate((qs_list, logits_prob))
            reads += predict_read
        val.pop(name)  # Release the memory
        basecall_time = time.time() - start_time
        bpreads = [index2base(read) for read in reads]
        js_ratio = FLAGS.jump/FLAGS.segment_len
        kernal = get_assembler_kernal(FLAGS.jump,FLAGS.segment_len)
        if FLAGS.extension == 'fastq':
            consensus, qs_consensus = simple_assembly_qs(bpreads, qs_list,js_ratio,kernal=kernal)
            qs_string = qs(consensus, qs_consensus)
        else:
            consensus = simple_assembly(bpreads,js_ratio,kernal=kernal)
        c_bpread = index2base(np.argmax(consensus, axis=0))
        assembly_time = time.time() - start_time
        list_of_time = [start_time, reading_time,
                        basecall_time, assembly_time]
        write_output(bpreads, c_bpread, list_of_time, file_pre, concise=FLAGS.concise, suffix=FLAGS.extension,
                     q_score=qs_string,global_setting=FLAGS)
    net.pbars.end()

def decoding_queue(logits_queue, num_threads=6):
    """
    Build the decoding queue graph.
    Args:
        logits_queue: the logits queue.
        num_threads: number of threads.
    Return:
        decode_predict: (decoded_sparse_tensor,decoded_probability)
            decoded_sparse_tensor is a [sparse tensor]
        decode_prob: a [batch_size] array contain the probability of each path.
        decode_fname: a [batch_size] array contain the filenames.
        decode_idx: a [batch_size] array contain the indexs.
        decodeedQueue.size(): The number of instances in the queue.
    """
    q_logits, q_name, q_index, seq_length = logits_queue.dequeue()
    batch_n = q_logits.get_shape().as_list()[0]
    if FLAGS.extension == 'fastq':
        prob = path_prob(q_logits)
    else:
        prob = tf.constant([0.0]*batch_n)  # We just need to have the right type, because of the queues
    if FLAGS.beam == 0:
        decode_decoded, decode_log_prob = tf.nn.ctc_greedy_decoder(tf.transpose(
            q_logits, perm=[1, 0, 2]), seq_length, merge_repeated=True)
    else:
        decode_decoded, decode_log_prob = tf.nn.ctc_beam_search_decoder(
            tf.transpose(q_logits, perm=[1, 0, 2]),
            seq_length, merge_repeated=False,
            beam_width=FLAGS.beam,top_paths = 1)  # There will be a second merge operation after the decoding process
        # if the merge_repeated for decode search decoder set to True.
        # Check this issue https://github.com/tensorflow/tensorflow/issues/9550
    decodeedQueue = tf.FIFOQueue(
        capacity=2 * num_threads,
        dtypes=[tf.int64 for _ in decode_decoded] * 3 + [tf.float32, tf.float32, tf.string, tf.int32],
    )
    ops = []
    for x in decode_decoded:
        ops.append(x.indices)
        ops.append(x.values)
        ops.append(x.dense_shape)
    decode_enqueue = decodeedQueue.enqueue(tuple(ops + [decode_log_prob, prob, q_name, q_index]))

    decode_dequeue = decodeedQueue.dequeue()
    decode_prob, decode_fname, decode_idx = decode_dequeue[-3:]

    decode_dequeue = decode_dequeue[:-3]
    decode_predict = [[], decode_dequeue[-1]]
    for i in range(0, len(decode_dequeue) - 1, 3):
        decode_predict[0].append(
            tf.SparseTensor(
                indices=decode_dequeue[i],
                values=decode_dequeue[i + 1],
                dense_shape=decode_dequeue[i + 2],
            )
        )

    decode_qr = tf.train.QueueRunner(decodeedQueue, [decode_enqueue]*num_threads)
    tf.train.add_queue_runner(decode_qr)
    return decode_predict, decode_prob, decode_fname, decode_idx, decodeedQueue.size()


def run(args):
    global FLAGS
    FLAGS = args
    # logging.debug("Flags:\n%s", pformat(vars(args)))
    time_dict = unix_time(evaluation)
    print(FLAGS.output)
    print('Real time:%5.3f Systime:%5.3f Usertime:%5.3f' %
          (time_dict['real'], time_dict['sys'], time_dict['user']))
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
    parser = argparse.ArgumentParser(prog='chiron',
                                     description='A deep neural network basecaller.')
    parser.add_argument('-i', '--input', required = True,
                        help="File path or Folder path to the fast5 file.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output Folder name")
    parser.add_argument('-m', '--model', required = True,
                        help="model folder path")
    parser.add_argument('-s', '--start', type=int, default=None,
                        help="Start index of the signal file.")
    parser.add_argument('-b', '--batch_size', type=int, default=None,
                        help="Batch size for run, bigger batch_size will increase the processing speed and give a slightly better accuracy but require larger RAM load")
    parser.add_argument('-l', '--segment_len', type=int, default=None,
                        help="Segment length to be divided into.")
    parser.add_argument('-j', '--jump', type=int, default=None,
                        help="Step size for segment")
    parser.add_argument('-t', '--threads', type=int, default=None,
                        help="Threads number")
    parser.add_argument('--beam', type=int, default=None,
                        help="Beam width used in beam search decoder, default is 0, in which a greedy decoder is used. Recommend width:100, Large beam width give better decoding result but require longer decoding time.")
    parser.add_argument('-e', '--extension', default='fastq',
                        help="Output file extension.")
    parser.add_argument('--concise', action='store_true',
                        help="Concisely output the result, the meta and segments files will not be output.")
    parser.add_argument('--mode', default = 'dna',
                        help="Output mode, can be chosen from dna or rna.")
    parser.add_argument('-p', '--preset',default=None,help="Preset evaluation parameters. Can be one of the following:\ndna-pre\nrna-pre")
    args = parser.parse_args(sys.argv[1:])
    def set_paras(p):
        args.start = p['start'] if args.start is None else args.start
        args.batch_size=p['batch_size'] if args.batch_size is None else args.batch_size
        args.segment_len=p['segment_len'] if args.segment_len is None else args.segment_len
        args.jump=p['jump'] if args.jump is None else args.jump
        args.threads=p['threads'] if args.threads is None else args.threads
        args.beam=p['beam'] if args.beam is None else args.beam
    if args.preset is None:
        default_p = {'start':0,'batch_size':400,'segment_len':500,'jump':490,'threads':0,'beam':30}
    elif args.preset == 'dna-pre':
        default_p = {'start':0,'batch_size':400,'segment_len':400,'jump':390,'threads':0,'beam':30}
        if args.mode=='rna':
            raise ValueError('Try to use the DNA preset parameter setting in RNA mode.')
    elif args.preset == 'rna-pre':
        default_p = {'start':0,'batch_size':300,'segment_len':2000,'jump':1900,'threads':0,'beam':30}
        if args.mode=='dna':
            raise ValueError('Attempt to use the RNA preset parameter setting in DNA mode, enable rna mode by --mode.')
    else:
        raise ValueError('Unknown presetting %s undifiend'%(args.preset))
    set_paras(default_p)
    run(args)
