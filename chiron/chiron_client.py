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

import sys
import threading
from chiron.utils.easy_assembler import simple_assembly_qs
from chiron.utils.easy_assembler import simple_assembly
from chiron.chiron_input import read_data_for_eval
from chiron.utils.raw import extract
from chiron.chiron_eval import sparse2dense, qs, index2base,write_output
# This is a placeholder for a Google-internal import.
import os
import grpc
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('batch_size', 1100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('raw_dir', '/tmp', 'Input raw signal directory. ')
tf.app.flags.DEFINE_string('output','/tmp/output',"Output data directory. ")
tf.app.flags.DEFINE_string('mode','dna','If basecalling in RNA mode or DNA mode.')
tf.app.flags.DEFINE_string('extension','fastq','output format, default is fastq')

FLAGS = tf.app.flags.FLAGS

class DNA_CONF():
    def __init__(self):
        self.SEGMENT_LEN = 400
        self.JUMP = 30
        self.START = 0
class RNA_CONF():
    def __init__(self):
        self.SEGMENT_LEN = 1000
        self.JUMP = 60
        self.START = 0
class _Result_Collection(object):
    def __init__(self):
        self.val = defaultdict(dict)
    def pop_out(self,f,n):
        """
        Pop out the n reads and delete the f in dict
        """
        reads = list()
        probs = list()
        for i in range(self.val[f]):
            reads.append(self.val[f][i]['predict'])
            probs.append(self.val[f][i]['logits_prob'])
        reads = reads[:n]
        probs = probs[:n]
        self.remove(f)
        return reads,probs
    def remove(self,file_name):
        self.val.pop(file_name)
def data_iterator(raw_dir,start = 0, segment_len = 400, jump_step = 30):
    for root,dirs,file_list in os.walk(raw_dir):
        for f in file_list:
            if not f.endswith('.signal'):
                continue
            f_p = os.path.join(root,f)
            data_set = read_data_for_eval(f_p, 
                                          step=jump_step, 
                                          seg_length=segment_len)
            reads_n = data_set.reads_n
            for i in range(0, reads_n, FLAGS.batch_size):
                batch_x, seq_len, _ = data_set.next_batch(
                    FLAGS.batch_size, shuffle=False, sig_norm=False)
                batch_x = np.pad(
                    batch_x, ((0, FLAGS.batch_size - len(batch_x)), (0, 0)), mode='constant')
                seq_len = np.pad(
                    seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='constant')
                yield batch_x,seq_len,i,f,len(range(0, reads_n, FLAGS.batch_size)),reads_n
def _post_process(collector, i, f):
    def _callback(result_future):
        predict = result_future.result().outputs['predict_sequences']
        log_prob = result_future.result().outputs['log_prob']
        predict = ([predict],log_prob)
        logits_prob = result_future.result().outputs['prob_logits']
        predict_read, uniq_list = sparse2dense(predict)
        predict_read = predict_read[0]
        uniq_list = uniq_list[0]
        logits_prob = logits_prob[uniq_list]
        collector.val['f'][i]['predict'] = predict_read
        collector.val['f'][i]['logits_prob'] = logits_prob
        exception = result_future.exception()
        if exception:
            print(exception)
        return _callback
def do_inference(hostport, work_dir, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.
    
    Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.
    
    Returns:
    The classification error rate.
    
    Raises:
    IOError: An error occurred processing test data set.
    """
    if FLAGS.mode == 'dna':
        CONF = DNA_CONF()
    elif FLAGS.mode == 'rna':
        CONF = RNA_CONF()
    else:
        raise ValueError("Mode has to be either rna or dna.")
    channel = grpc.insecure_channel(hostport)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'chiron'
    request.model_spec.signature_name = 'predict_seqeunces'
    collector = _Result_Collection()
    for batch_x,seq_len,i,f,N,reads_n in data_iterator(FLAGS.raw_dir):
        request.inputs['x'].CopyFrom(
            tf.contrib.util.make_tensor_proto(batch_x, shape=[FLAGS.batch_size, CONF.SEGMENT_LEN]))
        request.inputs['seq_length'].CopyFrom(
            tf.contrib.util.make_tensor_proto(seq_len, shape=[FLAGS.batch_size]))
        request.inputs['training'].CopyFrom(
            tf.contrib.util.make_tensor_proto(False, shape=[]))
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        result_future.add_done_callback(_post_process(collector,i,f))
        if len(collector.val[f]) == N:
            qs_string = None
            reads,probs = collector.pop_out(f,reads_n)
            bpreads = [index2base(read) for read in reads]
            consensus, qs_consensus = simple_assembly_qs(bpreads, probs)
            qs_string = qs(consensus, qs_consensus)
            c_bpread = index2base(np.argmax(consensus, axis=0))
            file_pre = os.path.splitext(f)[0]
            write_output(bpreads, 
                         c_bpread, 
                         [np.NaN]*4, 
                         file_pre, 
                         concise=False, 
                         suffix=FLAGS.extension,
                         q_score=qs_string)
def main(_):
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference()
  print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()
