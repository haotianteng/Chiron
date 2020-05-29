# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Mar 17 20:56:18 2018
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from distutils.version import StrictVersion
import tensorflow as tf
import json
import os
from chiron.cnn import getcnnfeature
from chiron.cnn import getcnnlogit
from chiron.rnn import rnn_layers
from chiron.rnn import rnn_layers_rna

MOVING_AVERAGE_DECAY = 0.9999
LR_BOUNDARY = [0.66,0.83]
LR_DECAY = [1e-1,1e-2]
MOMENTUM = 0.9
def save_model(config_path,configure):
    """Save the configuration to the config path
    Args:
        config_path(String) : path to save the configuration
        configure(Dict) : Key-value pair of the configuration
    """
    config_dir = config_path[:config_path.rindex(os.path.sep)]
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    with open(config_path,'w+') as config_file:
        json.dump(configure,config_file)
    return None

def read_config(config_file):
    if config_file is not None:
        config = json.load(open(config_file))
    else:
        config = {'cnn':{'model':'dna_model1'},
                  'rnn':{'layer_num':3,
                         'hidden_num':100,
                         'cell_type':'LSTM',
                         'layer_type':'normal'},
                  'opt_method':'Adam',
                  'fl_gamma': 2}
    return config

def loss(logits, seq_len, label, fl_gamma = 0):
    """Calculate a CTC loss from the input logits and label.

    Args:
        logits: Tensor of shape [batch_size,max_time,class_num], logits from last layer of the NN, usually from a
            Fully connected layyer.
        seq_len: Tensor of shape [batch_size], sequence length for each sample in the batch.
        label: A Sparse Tensor of labels, sparse tensor of the true label.
        fl_gamma: The gamma parameter of the focal loss(see paper: https://arxiv.org/pdf/1708.02002.pdf), if 0 then no focal loss is enabled.
    Returns:
        Tensor of shape [batch_size], losses of the batch.
    """
    loss = tf.nn.ctc_loss(label, logits, seq_len, ctc_merge_repeated=True,
                       time_major=False,ignore_longer_outputs_than_inputs=True)
    if fl_gamma > 0:
        if StrictVersion(tf.__version__) >= StrictVersion('1.11.0'):
            loss = tf.math.pow(1-tf.math.exp(-loss),fl_gamma)*loss
        else:
            loss = tf.pow(1-tf.exp(-loss),fl_gamma)*loss
#    tf.summary.histogram('loss_histogram', loss)
    loss = tf.reduce_mean(loss)
    tf.add_to_collection('losses',loss)
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss', loss)
    return tf.add_n(tf.get_collection('losses'),name = 'total_loss')


def train_opt(init_rate,max_steps,global_step=None, opt_name="Adam"):
    """Generate training op

    Args:
        init_rate: initial learning rate.
        max_steps: maximum training steps.
        global_step: A optional Scalar tensor, the global step recorded, if None no global stop will be recorded.
        opt_name: An optional string from : "Adam","SGD","RMSProp","Momentum". Defaults to "Adam". Specify the optimizer.
    Returns:
        opt: Optimizer
    """
    optimizers = {"Adam": tf.train.AdamOptimizer,
                  "SGD": tf.train.GradientDescentOptimizer,
                  "RMSProp": tf.train.RMSPropOptimizer,
                  "Momentum": tf.train.MomentumOptimizer}
    boundaries = [int(max_steps*LR_BOUNDARY[0]), int(max_steps*LR_BOUNDARY[1])]
    values = [init_rate * decay for decay in [1,LR_DECAY[0],LR_DECAY[1]]]
    learning_rate = tf.train.piecewise_constant(global_step,boundaries,values)
    if opt_name == "Momentum":
        opt = optimizers[opt_name](learning_rate,momentum = MOMENTUM,use_nesterov = True)
    else:
        opt = optimizers[opt_name](learning_rate)
    return opt

def prediction(logits, seq_length, label,beam_width = 30, top_paths=1):
    """
    Args:
        logits:Input logits from a RNN.Shape = [batch_size,max_time,class_num]
        seq_length:sequence length of logits. Shape = [batch_size]
        label:Sparse tensor of label.
        beam_width(Int):Beam width used in beam search decoder.
        top_paths:The number of top score path to choice from the decorder.
    """
    logits = tf.transpose(logits, perm=[1, 0, 2])
    if beam_width == 0:
        predict = tf.nn.ctc_greedy_decoder(
                                        logits, 
                                        seq_length, 
                                        merge_repeated=True)
    else:
        predict = tf.nn.ctc_beam_search_decoder(
                                        logits, 
                                        seq_length, 
                                        merge_repeated=False,
                                        top_paths=top_paths,
                                        beam_width=beam_width)
    predict = predict[0]
    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d, axis=0)
    d_min = tf.reduce_min(edit_d, axis=0)
    error = tf.reduce_mean(d_min, axis=0)
    tf.summary.scalar('Error_rate', error)
    return error,d_min,predict

def inference(x,sequence_len,training,full_sequence_len,configure, apply_ratio = False):
    """Infer a logits of the input signal batch.

    Args:
        x: Tensor of shape [batch_size, max_time,channel], a batch of the input signal with a maximum length `max_time`.
        sequence_len: Tensor of shape [batch_size], given the real lenghs of the segments.
        training: Placeholder of Boolean, Ture if the inference is during training.
        full_sequence_len: Scalar float, the maximum length of the sample in the batch.
        configure:Model configuration.
        apply_ratio: If apply the ration to the sequence_len or not.
    Returns:
        logits: Tensor of shape [batch_size, max_time, class_num]
        ratio: Scalar float, the scale factor between the output logits and the input maximum length.
    """
    print("DEBUG MODEL X Shape")
    print(x.shape)
    cnn_feature = getcnnfeature(x,training = training,cnn_config = configure['cnn'])
    feashape = cnn_feature.get_shape().as_list()
    ratio = full_sequence_len/feashape[1]
    if apply_ratio:
        sequence_len =  tf.cast(tf.ceil(tf.cast(sequence_len,tf.float32)/ratio),tf.int32)
    if configure['rnn']['layer_num'] == 0:
        logits = getcnnlogit(cnn_feature)
    elif configure['rnn']['layer_type'] == 'rna':
        logits = rnn_layers_rna(cnn_feature,
                            sequence_len,
                            training,
                            layer_num = configure['rnn']['layer_num'],
                            hidden_num = configure['rnn']['hidden_num'],
                            cell=configure['rnn']['cell_type'])
    else:
        logits = rnn_layers(cnn_feature,
                            sequence_len,
                            training,
                            layer_num = configure['rnn']['layer_num'],
                            hidden_num = configure['rnn']['hidden_num'],
                            cell=configure['rnn']['cell_type'])
        #logits = cudnn_rnn(cnn_feature,rnn_layer_num)
    return logits,ratio
