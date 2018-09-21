# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Tue Aug  8 08:31:18 2017
import os
import argparse
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants as sig_constants
from chiron.chiron_model import inference,read_config
from chiron.chiron_eval import path_prob

parser = argparse.ArgumentParser(description='Basecall a signal file')
parser.add_argument('-m', '--model', required = True, help="model folder")
parser.add_argument('-o', '--output_dir',required = True, help = "output folder")
parser.add_argument('-v', '--version', required = True, help = "Verison of the model")
parser.add_argument('-l', '--segment_len', type=int, default=400, help="Segment length to be divided into.")
parser.add_argument('-w','--beam_width',type = int,default = 30 , help="The window width of beam.")
FLAGS = parser.parse_args()


def output_list(x,seq_length):
    training = tf.constant(False,dtype = tf.bool,name = 'Training')
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
    return predict[0],logits,prob_logits,log_prob


def build_and_run_exports():
    """Given the latest checkpoint file export the saved model.
    """

    prediction_graph = tf.Graph()
    
    with prediction_graph.as_default():
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {'combined_input': tf.FixedLenFeature(shape=[FLAGS.segment_len+1], dtype=tf.float32)}
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        combined_input = tf.identity(tf_example['combined_input'],name = 'combined_input')
        x,seq_length = tf.split(combined_input,
                                num_or_size_splits = [FLAGS.segment_len,1],
                                axis = 1)
        seq_length_r = tf.reshape(seq_length,[tf.shape(seq_length)[0]])
        seq_len_32 = tf.cast(seq_length_r,dtype = tf.int32)
        #Inference
        predict,logits,prob_logits,log_prob = output_list(x,seq_len_32)
        values, indices = tf.nn.top_k(logits,k=1)
        saver = tf.train.Saver()

        with tf.Session(graph=prediction_graph) as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.model)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                print('Successfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
            else:
                print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
                return
            output_path = os.path.join(
                    tf.compat.as_bytes(FLAGS.output_dir),
                    tf.compat.as_bytes(str(FLAGS.version)))
            exporter = tf.saved_model.builder.SavedModelBuilder(output_path)
            # Build the signature_def_map.
            classification_inputs = tf.saved_model.utils.build_tensor_info(
                    serialized_tf_example)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
                    indices)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)
            classification_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs={tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs},
                outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
                  },
                method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))     
                
            # Build the predict_def_map
            input_tensor_info = tf.saved_model.utils.build_tensor_info(
                    combined_input)
            indices_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                    predict.indices)
            values_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                    predict.values)
            dense_shape_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                    predict.dense_shape)
            logits_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                    logits)
            prob_logits_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                    prob_logits)
            log_prob_output_tensor_info = tf.saved_model.utils.build_tensor_info(
                    log_prob)

            prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
                inputs = {'combined_inputs':input_tensor_info},
                outputs = {'indices':indices_output_tensor_info,
                           'values':values_output_tensor_info,
                           'dense_shape':dense_shape_output_tensor_info,
                           'logits':logits_output_tensor_info,
                           'prob_logits':prob_logits_output_tensor_info,
                           'log_prob':log_prob_output_tensor_info},
                method_name=sig_constants.PREDICT_METHOD_NAME))
            
            exporter.add_meta_graph_and_variables(
                sess,
                tags=[tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predicted_sequences':prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature,
                },
                main_op=tf.tables_initializer(),
                strip_default_attrs=True)
            exporter.save()


if __name__ == '__main__':
    build_and_run_exports()
