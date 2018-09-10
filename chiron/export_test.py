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

from chiron.chiron_model import inference,read_config
from chiron.chiron_eval import path_prob

parser = argparse.ArgumentParser(description='Basecall a signal file')
# parser.add_argument('-i','--input', help="File path or Folder path to the signal file.")
# parser.add_argument('-o','--output', help = "Output Folder name")
parser.add_argument('-m', '--model', required = True, help="model folder")
parser.add_argument('-b', '--batch_size', type=int, default=1100,
                    help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
parser.add_argument('-l', '--segment_len', type=int, default=400, help="Segment length to be divided into.")
parser.add_argument('-w','--beam_width',type = int,default = 30 , help="The window width of beam.")
FLAGS = parser.parse_args()


def input_output_list():
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    config_path = os.path.join(FLAGS.model,'model.json')
    model_configure = read_config(config_path)
    logits, ratio = inference(x, 
                              seq_length, 
                              training=training,
                              full_sequence_len = FLAGS.segment_len,
                              configure = model_configure)
    ratio = tf.constant(ratio,dtype = tf.float32,shape = [])
    seq_length_r = tf.cast(tf.round(tf.cast(seq_length,dtype = tf.float32)/ratio),tf.int32)
    prob_logits = path_prob(logits)
    predict,log_prob = tf.nn.ctc_beam_search_decoder(tf.transpose(logits, perm=[1, 0, 2]), 
                                            seq_length_r, 
                                            merge_repeated=True,
                                            beam_width = FLAGS.beam_width)
    input_dict = {'x': x, 'seq_length': seq_length, 'training': training}
    output_dict = {'predict_index':predict[0],'logits':logits, 'prob_logits':prob_logits,'log_prob':log_prob}
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
            for name, tensor in inputs_dict.items()
        }
        output_info = {
            name: tf.saved_model.utils.build_tensor_info(tensor)
            for name, tensor in prediction_dict.items()
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
