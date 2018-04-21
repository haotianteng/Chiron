# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Tue Aug  8 08:31:18 2017
import os

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants as sig_constants
import argparse

from chiron.chiron_eval import inference

parser = argparse.ArgumentParser(description='Basecall a signal file')
# parser.add_argument('-i','--input', help="File path or Folder path to the signal file.")
# parser.add_argument('-o','--output', help = "Output Folder name")
parser.add_argument('-m', '--model', default='../model/crnn3+3_S10_2_re', help="model folder")
parser.add_argument('-b', '--batch_size', type=int, default=1100,
                    help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
parser.add_argument('-l', '--segment_len', type=int, default=300, help="Segment length to be divided into.")
FLAGS = parser.parse_args()


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


def build_and_run_exports(job_dir, serving_input_fn):
    """Given the latest checkpoint file export the saved model.
    Args:
      job_dir (string): Location of checkpoints and model files
    """

    prediction_graph = tf.Graph()
    exporter = tf.saved_model.builder.SavedModelBuilder(
        os.path.join(job_dir, 'export'))
    with prediction_graph.as_default():
        inputs_dict, prediction_dict = serving_input_fn()
        saver = tf.train.Saver()

        inputs_info = {
            name: tf.saved_model.utils.build_tensor_info(tensor)
            for name, tensor in inputs_dict.iteritems()
        }
        output_info = {
            name: tf.saved_model.utils.build_tensor_info(tensor)
            for name, tensor in prediction_dict.iteritems()
        }
        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs_info,
            outputs=output_info,
            method_name=sig_constants.PREDICT_METHOD_NAME
        )

    with tf.Session(graph=prediction_graph) as session:
        session.run([tf.local_variables_initializer(), tf.tables_initializer()])
        saver.restore(session, tf.train.latest_checkpoint(job_dir))
        exporter.add_meta_graph_and_variables(
            session,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                sig_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
            },
            legacy_init_op=tf.saved_model.main_op.main_op()
        )

    exporter.save()


if __name__ == '__main__':
    build_and_run_exports(FLAGS.model, input_output_list)
