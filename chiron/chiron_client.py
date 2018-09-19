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
import grpc
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow_serving.apis import predict_pb2
#from tensorflow_serving.apis import prediction_service_pb2 as prediction_service_pb2_grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('batch_size', 1, 'Number of test signals')
tf.app.flags.DEFINE_string('server', '0.0.0.0:8500', 'PredictionService host:port')
tf.app.flags.DEFINE_string('raw_dir', '/home/heavens/Chiron_project/Chiron/chiron/example_data/DNA/output/raw/', 'Input raw signal directory. ')
tf.app.flags.DEFINE_string('output','/home/heavens/Chiron_project/Chiron/chiron/example_data/DNA/output/',"Output data directory. ")
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
        self.SEGMENT_LEN = 500
        self.JUMP = 50
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
        exception = result_future.exception()
        if exception:
            print(exception)
        else:
            indices = result_future.result().outputs['indices']
            values = result_future.result().outputs['values']
            log_prob = result_future.result().outputs['log_prob']
            logits_prob = result_future.result().outputs['prob_logits']
#            predict_read, uniq_list = sparse2dense(indices,values,log_prob)
#            predict_read = predict_read
#            uniq_list = uniq_list
#            logits_prob = logits_prob[uniq_list]
#            collector.val['f'][i]['predict'] = predict_read
#            collector.val['f'][i]['logits_prob'] = logits_prob
        
    return _callback
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
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'chiron'
    request.model_spec.signature_name = 'predicted_sequences'
    collector = _Result_Collection()
    for batch_x,seq_len,i,f,N,reads_n in data_iterator(FLAGS.raw_dir):
        seq_len = np.reshape(seq_len,(seq_len.shape[0],1))
        combined_input = np.concatenate((batch_x,seq_len),axis = 1).astype(np.float32)
        request.inputs['combined_inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto(combined_input, shape=[FLAGS.batch_size, CONF.SEGMENT_LEN+1]))
        result_future = stub.Predict.future(request, 5.0)  # 5 seconds
        print(result_future.result().outputs['indices'])
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
        break
        
def main():
    if not FLAGS.server:
        print('please specify server host:port')
        return
    result = do_inference()


if __name__ == '__main__':
    main()