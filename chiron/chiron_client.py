# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.

The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

from chiron.utils.easy_assembler import simple_assembly_qs
from chiron.chiron_input import read_data_for_eval
from chiron.chiron_eval import qs, index2base,write_output
# This is a placeholder for a Google-internal import.
import os
import sys
import time
import grpc
import argparse
import numpy as np
import tensorflow as tf
import threading
from collections import defaultdict
from tensorflow_serving.apis import predict_pb2
#from tensorflow_serving.apis import prediction_service_pb2 as prediction_service_pb2_grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc
from chiron.utils.progress import multi_pbars

NEST_DICT = lambda: defaultdict(NEST_DICT)
class DNA_CONF():
    def __init__(self):
        self.SEGMENT_LEN = 400
        self.JUMP = 30
        self.START = 0
class RNA_CONF():
    def __init__(self):
        self.SEGMENT_LEN = 500
        self.JUMP = 50
        self.START = 0
class _Result_Collection(object):
    def __init__(self,concurrency):
        self.val = NEST_DICT()
        self.batch_n = dict()
        self.reads_n = dict()
        self._condition = threading.Condition()
        self._concurrency = concurrency
        self._error = 0
        self._done = list()
        self._active = 0
        self._begin = False
    def pop_out(self,f):
        """
        Pop out the n reads and delete the f in dict
        """
        reads = list()
        probs = np.empty((0, 1), dtype=np.float)
        for i in range(len(self.val[f])):
            reads+=self.val[f][i]['predict']
            probs = np.concatenate((probs,self.val[f][i]['logits_prob']))
        reads = reads[:self.reads_n[f]]
        probs = probs[:self.reads_n[f]]
        self.remove(f)
        return reads,probs
    def remove(self,file_name):
        self.val.pop(file_name)
        self.batch_n.pop(file_name)
        self._done.remove(file_name)
    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self,f):
        with self._condition:
          self._done.append(f)
          self._condition.notify()
    
    def dec_active(self):
        with self._condition:
          self._active -= 1
          self._condition.notify()
    
    def all_done(self):
        with self._condition:
          return ((self._active ==0) and (len(self.val) == 0) and (self._begin))
        
    def throttle(self):
        with self._condition:
          while self._active == self._concurrency:
            self._condition.wait()
          self._active += 1
          self._begin = True
def sparse2dense(indices,values,log_probability):
    """Transfer a sparse input in to dense representation
    Args:
        indices: Matrix of shape [total_decoded_outputs[j], 2], each row stand for [batch, time] index in dense representation.
        values: Vector of shape [total_decoded_outputs[j]]. The vector stores the decoded classes for beam j.
        log_probability: A float matrix of shape [batch_size, top_paths]. Give the sequence log-probabilities.

    Returns:
        predict_read[Float]: Nested List, [path_index][read_index][base_index], give the list of decoded reads(in number representation 0-A, 1-C, 2-G, 3-T).
        uniq_list[Int]: Nested List, [top_paths][batch_index], give the batch index that exist in the decoded output.
    """

    predict_read = list()
    unique, pre_counts = np.unique(
        indices[:, 0], return_counts=True)
    pos_predict = 0
    for indx, _ in enumerate(pre_counts):
        predict_read.append(
            values[pos_predict:pos_predict + pre_counts[indx]])
        pos_predict += pre_counts[indx]
    return predict_read, unique
def gen_file_list(input):
    total_file_list = list()
    for root,dirs,file_list in os.walk(input):
        for f in file_list:
            if f.endswith('.signal'):
                f_p = os.path.join(root,f)
                total_file_list.append(f_p)
    return total_file_list
def data_iterator(file_list,start = 0, segment_len = 400, jump_step = 30):
    for f_p in file_list:
        data_set = read_data_for_eval(f_p, 
                                      step=jump_step, 
                                      seg_length=segment_len)
        reads_n = data_set.reads_n
        index = -1
        for i in range(0, reads_n, FLAGS.batch_size):
            batch_x, seq_len, _ = data_set.next_batch(
                FLAGS.batch_size, shuffle=False, sig_norm=False)
            batch_x = np.pad(
                batch_x, 
                ((0, FLAGS.batch_size - len(batch_x)), (0, 0)),
                mode='constant')
            seq_len = np.pad(
                seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='constant')
            index += 1
            yield batch_x,seq_len,index,f_p,len(range(0, reads_n, FLAGS.batch_size)),reads_n
                
def _post_process(collector, i, f,N,reads_n):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            indices = tf.make_ndarray(result_future.result().outputs['indices'])
            values = tf.make_ndarray(result_future.result().outputs['values'])
            log_prob = tf.make_ndarray(result_future.result().outputs['log_prob'])
            logits_prob = tf.make_ndarray(result_future.result().outputs['prob_logits'])
            predict_read, uniq_list = sparse2dense(indices,values,log_prob)
            uniq_list = uniq_list
            logits_prob = logits_prob[uniq_list]
            collector.val[f][i]['predict'] = predict_read
            collector.val[f][i]['logits_prob'] = logits_prob
            if f not in collector.batch_n.keys():
                collector.batch_n[f] = N
                collector.reads_n[f] = reads_n
            collector.dec_active()
            if len(collector.val[f]) >= N:
                collector.inc_done(f)
                collector.reads_n[f] = reads_n
    return _callback
def make_dirs(output):
    if not os.path.exists(output):
            os.makedirs(output)
    if not os.path.exists(os.path.join(output, 'segments')):
        os.makedirs(os.path.join(output, 'segments'))
    if not os.path.exists(os.path.join(output, 'result')):
        os.makedirs(os.path.join(output, 'result'))
    if not os.path.exists(os.path.join(output, 'meta')):
        os.makedirs(os.path.join(output, 'meta'))
def do_inference():
    """Tests PredictionService with concurrent requests.    
    Raises:
    IOError: An error occurred processing test data set.
    """
    if FLAGS.mode == 'dna':
        CONF = DNA_CONF()
    elif FLAGS.mode == 'rna':
        CONF = RNA_CONF()
    else:
        raise ValueError("Mode has to be either rna or dna.")
    make_dirs(FLAGS.output)
    FLAGS.segment_len = CONF.SEGMENT_LEN
    FLAGS.jump = CONF.JUMP
    FLAGS.start = CONF.START
    pbars = multi_pbars(["Request Submit:","Request finished"])
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'chiron'
#    request.model_spec.signature_name = 'predicted_sequences'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    collector = _Result_Collection(concurrency = FLAGS.concurrency)
    file_list = gen_file_list(FLAGS.input)
    batch_iterator = data_iterator(file_list)
    def submit_fn():
        for batch_x,seq_len,i,f,N,reads_n in batch_iterator:
            seq_len = np.reshape(seq_len,(seq_len.shape[0],1))
#            combined_input = np.concatenate((batch_x,seq_len),axis = 1).astype(np.float32)
#            request.inputs['combined_inputs'].CopyFrom(
#                tf.contrib.util.make_tensor_proto(combined_input, shape=[FLAGS.batch_size, CONF.SEGMENT_LEN+1]))
            request.inputs['x'].CopyFrom(tf.contrib.util.make_tensor_proto(batch_x, 
                         shape=[FLAGS.batch_size, CONF.SEGMENT_LEN]))
            request.inputs['seq_len'].CopyFrom(tf.contrib.util.make_tensor_proto(seq_len,
                         shape=[FLAGS.batch_size]))
            collector.throttle()
            result_future = stub.Predict.future(request, 100.0)  # 5 seconds
            result_future.add_done_callback(_post_process(collector,i,f,N,reads_n))
            pbars.update(0,total = reads_n,progress = (i+1)*FLAGS.batch_size)
            pbars.update_bar()
    submiter = threading.Thread(target=submit_fn,args=())
    submiter.setDaemon(True)
    submiter.start()
    pbars.update(1,total = len(file_list))
    pbars.update_bar()
    while not collector.all_done():
        if len(collector._done) > 0:
            qs_string = None
            f_p = collector._done[0]
            reads,probs = collector.pop_out(f_p)
            bpreads = [index2base(read) for read in reads]
            consensus, qs_consensus = simple_assembly_qs(bpreads, probs)
            qs_string = qs(consensus, qs_consensus)
            c_bpread = index2base(np.argmax(consensus, axis=0))
            file_pre = os.path.basename(os.path.splitext(f_p)[0])
            write_output(bpreads, 
                         c_bpread, 
                         [np.NaN]*4, 
                         file_pre, 
                         concise=FLAGS.concise, 
                         suffix=FLAGS.extension,
                         q_score=qs_string,
                         global_setting = FLAGS)
            pbars.update(1,progress = pbars.progress[1]+1)
            pbars.update_bar()
        
def main():
    if not FLAGS.server:
        print('please specify server host:port')
        return
    do_inference()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='chiron',
                                     description='A deep neural network basecaller.')
    parser.add_argument('-i', '--input', required = True,
                        help="File path or Folder path to the signal file.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output Folder name")
    parser.add_argument('-b', '--batch_size', type=int, default=1100,
                        help="Batch size for run, bigger batch_size will increase the processing speed and give a slightly better accuracy but require larger RAM load")
    parser.add_argument('-t', '--concurrency', type=int, default=1,
                        help="Threads number")
    parser.add_argument('-e', '--extension', default='fastq',
                        help="Output file extension.")
    parser.add_argument('--concise', action='store_true',
                        help="Concisely output the result, the meta and segments files will not be output.")
    parser.add_argument('--mode', default = 'dna',
                        help="Output mode, can be chosen from dna or rna.")
    parser.add_argument('--server',default = '0.0.0.0:8500',help = 'PredictionService host:port')
    FLAGS = parser.parse_args(sys.argv[1:])
    FLAGS.model = "chiron_serving"
    main()