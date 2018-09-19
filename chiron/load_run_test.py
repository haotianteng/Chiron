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

import chiron.chiron_model as chiron_model
from chiron.chiron_model import inference
from chiron.chiron_input import read_data_for_eval
from tensorflow.contrib import predictor
from google.protobuf import text_format
from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils

def input_output_list():
    x = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    model_configure = chiron_model.read_config(FLAGS.config_path)
    logits, _ = inference(x, seq_length, training=training,full_sequence_len = FLAGS.segment_len, configure = model_configure)
    predict = tf.nn.ctc_greedy_decoder(tf.transpose(logits, perm=[1, 0, 2]), seq_length, merge_repeated=True)
    input_dict = {'x': x, 'seq_length': seq_length, 'training': training}
    output_dict = {'decoded_indices': predict[0][0].indices, 'decoded_values': predict[0][0].values,
                   'neg_sum_logits': predict[1]}
    return input_dict, output_dict


# if __name__=='__main__':
#parser = argparse.ArgumentParser()
#parser.add_argument("--export_folder", default="/home/heavens/Chiron_project/Chiron/chiron/model/DNA_default/export_BAK/1/", type=str,
#                    help="Frozen model file to import")
#parser.add_argument('-m', '--model', default='../model/crnn3+3_S10_2_re', help="model folder")
#parser.add_argument('-b', '--batch_size', type=int, default=1100,
#                    help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
#parser.add_argument('-l', '--segment_len', type=int, default=400, help="Segment length to be divided into.")
#parser.add_argument('-s', '--start', type=int, default=0, help="Start index of the signal file.")
#parser.add_argument('-j', '--jump', type=int, default=30, help="Step size for segment")
#parser.add_argument('-t', '--threads', type=int, default=0, help="Threads number")
#parser.add_argument('--config_path',default = '/home/heavens/Chiron_project/Chiron/chiron/model/DNA_default/model.json')
#FLAGS = parser.parse_args()
#with tf.Session() as sess:
#    meta_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.export_folder)
#    graph = tf.get_default_graph()
#    predict_fn = predictor.from_saved_model(FLAGS.export_folder)
#    input_path = '/home/heavens/Chiron_project/Chiron/chiron/example_data/DNA/output/raw/read1.signal'
#    eval_data = read_data_for_eval(input_path, FLAGS.start, seg_length=FLAGS.segment_len, step=FLAGS.jump)
#    batch_x, seq_len, _ = eval_data.next_batch(FLAGS.batch_size, shuffle=False)
#    batch_x = np.pad(batch_x, ((0, FLAGS.batch_size - len(batch_x)), (0, 0)), mode='constant')
#    seq_len = np.pad(seq_len, ((0, FLAGS.batch_size - len(seq_len))), mode='constant')
##    predictions = predict_fn({'signals':batch_x,'seq_len':seq_len})
#    seq_len = np.reshape(seq_len,(seq_len.shape[0],1))
#    combined_input = np.concatenate((batch_x,seq_len),axis = 1)
#    predictions = predict_fn({'inputs':combined_input})
#    input_dict, output_dict = input_output_list()
#    feed_dict = {input_dict['x']:batch_x,input_dict['seq_length']:seq_len,input_dict['training']:False}
##    sess.run(output_dict['decoded_indices'],feed_dict=feed_dict)

def get_meta_graph_def(saved_model_dir, tags):
  """Gets `MetaGraphDef` from a directory containing a `SavedModel`.
  Returns the `MetaGraphDef` for the given tag-set and SavedModel directory.
  Args:
    saved_model_dir: Directory containing the SavedModel.
    tags: Comma separated list of tags used to identify the correct
      `MetaGraphDef`.
  Raises:
    ValueError: An error when the given tags cannot be found.
  Returns:
    A `MetaGraphDef` corresponding to the given tags.
  """
  saved_model = reader.read_saved_model(saved_model_dir)
  set_of_tags = set([tag.strip() for tag in tags.split(',')])
  for meta_graph_def in saved_model.meta_graphs:
    if set(meta_graph_def.meta_info_def.tags) == set_of_tags:
      return meta_graph_def
  raise ValueError('Could not find MetaGraphDef with tags {}'.format(tags))
  
exp_dir = "/home/heavens/Chiron_project/Chiron/chiron/model/DNA_default/export/1/"
metagraph_def = get_meta_graph_def(exp_dir,'serve')
signature_def = signature_def_utils.get_signature_def_by_key(
        metagraph_def,
        'predicted_sequences')
output_names = {k: v.name for k, v in signature_def.outputs.items()}
print(output_names)
predict_fn = predictor.from_saved_model(exp_dir)
seq_len = np.asarray([[4]])
combined_input = np.concatenate((batch_x,seq_len),axis = 1)
predictions = predict_fn({'inputs':combined_input})