# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Fri Apr 21 03:46:35 2017
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn.python.ops.rnn import stack_bidirectional_dynamic_rnn
from chiron.utils.lstm import BNLSTMCell
from chiron.utils.variable import _variable_on_cpu

def rnn_layers(x, 
               seq_length, 
               training, 
               hidden_num=100, 
               layer_num=3, 
               class_n=5, 
               cell='LSTM',
               dtype = tf.float32):
    """Generate RNN layers.

    Args:
        x (Float): A 3D-Tensor of shape [batch_size,max_time,channel]
        seq_length (Int): A 1D-Tensor of shape [batch_size], real length of each sequence.
        training (Boolean): A 0D-Tenosr indicate if it's in training.
        hidden_num (int, optional): Defaults to 100. Size of the hidden state, 
            hidden unit will be deep concatenated, so the final hidden state will be size of 200.
        layer_num (int, optional): Defaults to 3. Number of layers in RNN.
        class_n (int, optional): Defaults to 5. Number of output class.
        cell(str): A String from 'LSTM','GRU','BNLSTM', the RNN Cell used. 
            BNLSTM stand for Batch normalization LSTM Cell.

    Returns:
         logits: A 3D Tensor of shape [batch_size, max_time, class_n]
    """

    cells_fw = list()
    cells_bw = list()
    for i in range(layer_num):
        if cell == 'LSTM':
            cell_fw = LSTMCell(hidden_num)
            cell_bw = LSTMCell(hidden_num)
        elif cell == 'GRU':
            cell_fw = GRUCell(hidden_num)
            cell_bw = GRUCell(hidden_num)
        elif cell == 'BNLSTM':
            cell_fw = BNLSTMCell(hidden_num,training = training)
            cell_bw = BNLSTMCell(hidden_num,training = training)
        else:
            raise ValueError("Cell type unrecognized.")
        cells_fw.append(cell_fw)
        cells_bw.append(cell_bw)
    multi_cells_fw = tf.nn.rnn_cell.MultiRNNCell(cells_fw)
    multi_cells_bw = tf.nn.rnn_cell.MultiRNNCell(cells_bw)
    with tf.variable_scope('BDGRU_rnn') as scope:
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=multi_cells_fw, cell_bw=multi_cells_bw, inputs=x, sequence_length=seq_length, dtype=dtype, scope=scope)
        lasth = tf.concat(outputs, 2, name='birnn_output_concat')
    # shape of lasth [batch_size,max_time,hidden_num*2]
    batch_size = lasth.get_shape().as_list()[0]
    max_time = lasth.get_shape().as_list()[1]
    with tf.variable_scope('rnn_fnn_layer'):
        weight_out = _variable_on_cpu(name='weights',
                                      shape = [2, hidden_num],
                                      initializer = tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / (2*hidden_num))),
                                      dtype = dtype)
        biases_out = _variable_on_cpu(name='bias',
                                      shape = [hidden_num],
                                      initializer = tf.zeros_initializer(),
                                      dtype = dtype)
        weight_class = _variable_on_cpu(name='weights_class',
                                        shape = [hidden_num, class_n],
                                        initializer = tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / hidden_num)),
                                        dtype = dtype)
        bias_class = _variable_on_cpu(name='bias_class',
                                      shape = [class_n],
                                      initializer = tf.zeros_initializer(),
                                      dtype = dtype)
        lasth_rs = tf.reshape(
            lasth, [batch_size, max_time, 2, hidden_num], name='lasth_rs')
        lasth_output = tf.nn.bias_add(tf.reduce_sum(tf.multiply(
            lasth_rs, weight_out), axis=2), biases_out, name='lasth_bias_add')
        lasth_output_rs = tf.reshape(
            lasth_output, [batch_size*max_time, hidden_num], name='lasto_rs')
        logits = tf.reshape(tf.nn.bias_add(tf.matmul(lasth_output_rs, weight_class), bias_class), [
                            batch_size, max_time, class_n], name="rnn_logits_rs")
    return logits


def rnn_layers_one_direction(x, seq_length, training, hidden_num=200, layer_num=3, class_n=5,dtype = tf.float32):
    """Create a single direction RNN layer

    Args:
        x (Float): Input 3D-Tensor of shape [batch_size, max_time, channel]
        seq_length (Int): A 1D-Tensor of shape [batch_size], real length of each sequence.
        training (Boolean): A 0D-Tenosr indicate if it's in training.
        hidden_num (int, optional): Defaults to 200. Size of the hidden state.
        layer_num (int, optional): Defaults to 3. Number of layers in RNN.
        class_n (int, optional): Defaults to 5. Number of output class.

    Returns:
        logits: A 3D Tensor of shape [batch_size, max_time, class_n]
    """

    cells = list()
    for i in range(layer_num):
        cell = BNLSTMCell(hidden_num, training)
        cells.append(cell)
    cell_wrap = tf.contrib.rnn.MultiRNNCell(cells)
    with tf.variable_scope('LSTM_rnn') as scope:
        lasth, _ = tf.nn.dynamic_rnn(
            cell_wrap, x, sequence_length=seq_length, dtype=tf.float32, scope=scope)
    # shape of lasth [batch_size,max_time,hidden_num*2]
    batch_size = lasth.get_shape().as_list()[0][type]
    max_time = lasth.get_shape().as_list()[1]
    with tf.variable_scope('rnn_fnn_layer'):
        weight_class = _variable_on_cpu(name='weights_class',
                                        shape = [hidden_num, class_n],
                                        initializer = tf.truncated_normal_initializer(stddev=np.sqrt(2.0 / hidden_num)),
                                        dtype = dtype)
        bias_class = _variable_on_cpu(name='bias_class',
                                      shape = [class_n],
                                      initializer = tf.zeros_initializer(),
                                      dtype = dtype)
        lasth_rs = tf.reshape(
            lasth, [batch_size * max_time, hidden_num], name='lasth_rs')
        logits = tf.reshape(tf.nn.bias_add(tf.matmul(lasth_rs, weight_class), bias_class),
                            [batch_size, max_time, class_n], name="rnn_logits_rs")
    return logits

def cudnn_rnn(x, layer_num, cell_num=100, training = True):
    """Create a cudnn fused RNN layer.
    
    Args:
        x (Float): Input 3D Tensor of shape [batch_size,max_time, channel]
        layer_num (Int): Number of layers in the RNN
        cell_num (Int, optional): Number of hidden cell. Default is 100.
        training (bool, optional): Defaults to True.
    """

    cnn_feature_shape = x.get_shape().as_list()
    rnn_cudnn = tf.contrib.cudnn_rnn.CudnnGRU(
                        layer_num, 
                        cell_num,
                        direction='bidirectional')
    logits,_ = rnn_cudnn(tf.transpose(x,perm = [1,0,2]))
    return tf.transpose(logits,[1,0,2])
