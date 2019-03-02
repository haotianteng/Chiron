# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Sat Apr 15 02:48:26 2017

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import tensorflow as tf
from chiron.utils.variable import _variable_on_cpu
from chiron.utils.variable import _variable_with_weight_decay
def conv_layer(indata, ksize, padding, training, name, dilate=1, strides=None, bias_term=False, active=True,
               BN=True, active_function='relu',wd = None):
    """A convolutional layer

    Args:
        indata: A input 4D-Tensor of shape [batch_size, Height, Width, Channel].
        ksize: A length 4 list.
        padding: A String from: "SAME","VALID"
        training: Scalar Tensor of type boolean, indicate if in training or not.
        name: A String give the name of this layer, other variables and options created in this layer will have this name as prefix.
        dilate (int, optional): Defaults to 1. Dilation of the width.
        strides (list, optional): Defaults to [1, 1, 1, 1]. A list of length 4.
        bias_term (bool, optional): Defaults to False. If True, a bais Tensor is added.
        active (bool, optional): Defaults to True. If True, output is activated by a activation function.
        BN (bool, optional): Defaults to True. If True, batch normalization will be applied. 
        active_function (str, optional): Defaults to 'relu'. A String from 'relu','sigmoid','tanh'.
        wd: weight decay, if None no weight decay will be added.

    Returns:
        conv_out: A output 4D-Tensor.
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    else:
        if type(strides) is int:
            strides = [1,1,strides,1]
    with tf.variable_scope(name):
        W = _variable_with_weight_decay("weights", 
                                        shape=ksize,
                                        wd=wd,
                                        initializer = tf.contrib.layers.xavier_initializer(uniform = False, ))
        if bias_term:
            b = _variable_on_cpu("bias", 
                                shape=[ksize[-1]],
                                initializer = tf.constant_initializer(0.0))
        if dilate > 1:
            if bias_term:
                conv_out = b + tf.nn.convolution(input=indata, filter=W, dilation_rate=np.asarray([1, dilate]),
                                                 padding=padding, name=name)
            else:
                conv_out = tf.nn.convolution(input=indata, filter=W, dilation_rate=np.asarray([1, dilate]),
                                             padding=padding, name=name)
        else:
            if bias_term:
                conv_out = b + \
                    tf.nn.conv2d(indata, W, strides=strides,
                                 padding=padding, name=name)
            else:
                conv_out = tf.nn.conv2d(
                    indata, W, strides=strides, padding=padding, name=name)
    if BN:
        with tf.variable_scope(name + '_bn') as scope:
            conv_out = batchnorm(conv_out,scope=scope,training = training)
#            conv_out = simple_global_bn(conv_out, name=name + '_bn')
            #conv_out = tf.layers.batch_normalization(conv_out,axis = -1,training = training,name = 'bn')
    if active:
        if active_function == 'relu':
            with tf.variable_scope(name + '_relu'):
                conv_out = tf.nn.relu(conv_out, name='relu')
        elif active_function == 'sigmoid':
            with tf.variable_scope(name + '_sigmoid'):
                conv_out = tf.sigmoid(conv_out, name='sigmoid')
        elif active_function == 'tanh':
            with tf.variable_scope(name + '_tanh'):
                conv_out = tf.tanh(conv_out, name='tanh')
        elif active_function == "elu":
            with tf.variable_scope(name + '_elu'):
                conv_out = tf.nn.elu(conv_out, name='elu')
    return conv_out

def gated_conv_layer(indata, kernal_width, out_channel, training):
    """
    Gated Convolutional layer
    Args:
        indata: input data.
        kernal_width: width of the kernal.
        out_channel: Number of out channel.
        padding: A string from: "SAME" "VALID", padding type
        training: If the convolutional network is called from training.
    """
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    gate_out = conv_layer(indata = indata, 
                          ksize = [1,kernal_width,in_channel, out_channel], 
                          padding = 'SAME', 
                          training = training, 
                          name = 'gate', 
                          bias_term = True, 
                          active = False)
    
    conv_out = conv_layer(indata = indata,
                          ksize = [1,kernal_width,in_channel, out_channel],
                          padding = 'SAME',
                          training = training,
                          name = 'conv',
                          bias_term = True,
                          active = 'tanh')
    out = gate_out * conv_out
    return out
def batchnorm(inp, scope, training, decay=0.99, epsilon=1e-5):
    """Applied batch normalization on the last axis of the tensor.

    Args:
        inp: A input Tensor
        scope: A string or tf.VariableScope.
        training (Boolean)): A scalar boolean tensor.
        decay (float, optional): Defaults to 0.99. The mean renew as follow: mean = pop_mean * (1- decay) + decay * old_mean
        epsilon (float, optional): Defaults to 1e-5. A small float number to avoid dividing by 0.

    Returns:
        The normalized, scaled, offset tensor.
    """

    with tf.variable_scope(scope):
        size = inp.get_shape().as_list()[-1]
        scale = tf.get_variable(
            'scale', shape=[size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', shape=[size])

        pop_mean = tf.get_variable(
            'pop_mean', shape=[size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable(
            'pop_var', shape=[size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(inp, [0, 1, 2])

        train_mean_op = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(inp, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(inp, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)


def simple_global_bn(inp, name):
    """Global batch normalization
    This tensor is nomalized by the global mean of the input tensor along the last axis.

    Args:
        inp : A 4D-Tensor.
        name (str): Name of the operation.

    Returns:
        global batch normalized tensor.
    """

    ksize = inp.get_shape().as_list()
    ksize = [ksize[-1]]
    mean, variance = tf.nn.moments(inp, [0, 1, 2], name=name + '_moments')
    scale = _variable_on_cpu(name + "_scale",
                            shape=ksize,
                            initializer=tf.contrib.layers.variance_scaling_initializer())
    offset = _variable_on_cpu(name + "_offset",
                             shape=ksize,
                             initializer=tf.contrib.layers.variance_scaling_initializer())
    return tf.nn.batch_normalization(inp, mean=mean, variance=variance, scale=scale, offset=offset,
                                     variance_epsilon=1e-5)


def inception_layer(indata, training, times=16):
    """Inception module with dilate conv layer from http://arxiv.org/abs/1512.00567
    Args:
        indata: A 4D-Tensor.
        training: Boolean.
        times: The base channel nubmer.
    Returns:
        A 4D-Tensor, the output of the inception layer.
    """

    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1_AvgPooling'):
        avg_pool = tf.nn.avg_pool(indata, ksize=(1, 1, 3, 1), strides=(1, 1, 1, 1), padding='SAME',
                                  name='avg_pool0a1x3')
        conv1a = conv_layer(avg_pool, ksize=[1, 1, in_channel, times * 3], padding='SAME', training=training,
                            name='conv1a_1x1')
    with tf.variable_scope('branch2_1x1'):
        conv0b = conv_layer(indata, ksize=[1, 1, in_channel, times * 3], padding='SAME', training=training,
                            name='conv0b_1x1')
    with tf.variable_scope('branch3_1x3'):
        conv0c = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME', training=training,
                            name='conv0c_1x1')
        conv1c = conv_layer(conv0c, ksize=[1, 3, times * 2, times * 3], padding='SAME', training=training,
                            name='conv1c_1x3')
    with tf.variable_scope('branch4_1x5'):
        conv0d = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME', training=training,
                            name='conv0d_1x1')
        conv1d = conv_layer(conv0d, ksize=[1, 5, times * 2, times * 3], padding='SAME', training=training,
                            name='conv1d_1x5')
    with tf.variable_scope('branch5_1x3_dilate_2'):
        conv0e = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME', training=training,
                            name='conv0e_1x1')
        conv1e = conv_layer(conv0e, ksize=[1, 3, times * 2, times * 3], padding='SAME', training=training,
                            name='conv1e_1x3_d2', dilate=2)
    with tf.variable_scope('branch6_1x3_dilate_3'):
        conv0f = conv_layer(indata, ksize=[1, 1, in_channel, times * 2], padding='SAME', training=training,
                            name='conv0f_1x1')
        conv1f = conv_layer(conv0f, ksize=[1, 3, times * 2, times * 3], padding='SAME', training=training,
                            name='conv1f_1x3_d3', dilate=3)
    return (tf.concat([conv1a, conv0b, conv1c, conv1d, conv1e, conv1f], axis=-1, name='concat'))


def residual_layer(indata, out_channel, training, i_bn=False,k = 3, strides = None):
    """An inplementation of the residual layer from https://arxiv.org/abs/1512.03385

    Args:
        indata: A 4-D Tensor
        out_channel (Int): The number of out channel
        training (Boolean): 0-D Boolean Tensor indicate if it's in training.
        i_bn (bool, optional): Defaults to False. If the identity layer being batch nomalized.

    Returns:
        relu_out: A 4-D Tensor of shape [batch_size, Height, Weight, out_channel]
    """

    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1'):
        indata_cp = conv_layer(indata, ksize=[1, 1, in_channel, out_channel], padding='SAME', training=training,
                               name='conv1', BN=i_bn, active=False,strides = strides)
    with tf.variable_scope('branch2'):
        conv_out1 = conv_layer(indata, ksize=[1, 1, in_channel, out_channel], padding='SAME', training=training,
                               name='conv2a', bias_term=False)
        conv_out2 = conv_layer(conv_out1, ksize=[1, k, out_channel, out_channel], padding='SAME', training=training,
                               name='conv2b', bias_term=False,strides = strides)
        conv_out3 = conv_layer(conv_out2, ksize=[1, 1, out_channel, out_channel], padding='SAME', training=training,
                               name='conv2c', bias_term=False, active=False)
    with tf.variable_scope('plus'):
        relu_out = tf.nn.relu(indata_cp + conv_out3, name='final_relu')
    return relu_out

def residual_layer_identity_mapping(indata, out_channel, training, k = 3, strides = None):
    """An inplementation of the residual layer from https://arxiv.org/abs/1512.03385

    Args:
        indata: A 4-D Tensor
        out_channel (Int): The number of out channel
        training (Boolean): 0-D Boolean Tensor indicate if it's in training.
        i_bn (bool, optional): Defaults to False. If the identity layer being batch nomalized.

    Returns:
        relu_out: A 4-D Tensor of shape [batch_size, Height, Weight, out_channel]
    """

    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1'):
        if strides is not None:
            indata_cp = tf.nn.avg_pool(indata,ksize = [1,1,strides,1],strides = [1,1,strides,1],padding = "SAME")
        else:
            indata_cp = indata
        if out_channel>in_channel:
            diff = out_channel - in_channel
            paddings = tf.constant([[0,0],[0,0],[0,0],[0,diff]])
            indata_cp = tf.pad(indata_cp,paddings = paddings,mode = 'CONSTANT',name = 'padding',constant_values = 0)
    with tf.variable_scope('branch2'):
        bn_1 = simple_global_bn(indata,name = 'pre_bn1')
        relu_1 = tf.nn.relu(bn_1,name = 'pre_relu_1')
        conv_1 = conv_layer(relu_1, ksize=[1, k, in_channel, out_channel], padding='SAME', training=training,
                               name='conv2a', bias_term=True,BN = False,strides = strides,active = False)
        bn_2 = simple_global_bn(conv_1,name = 'pre_bn2')
        relu_2 = tf.nn.relu(bn_2,name = 'pre_relu2')
        conv_2 = conv_layer(relu_2, ksize=[1, k, out_channel, out_channel], padding='SAME', training=training,
                               name='conv2b', bias_term=True,BN = False,strides = 1,active = False)
    return conv_2+indata_cp

def wavenet_layer(indata, out_channel, training, dilate, gated_activation=True, i_bn=True):
    """    An implementation of a variant of the Wavenet layer. https://arxiv.org/abs/1609.03499

    Args:
        indata: A 4D-Tensor.
        out_channel (Int): The number of output channel.
        training (Boolean): A scalar Boolean Tensor.
        dilate (Int): Dilation rate of the width.
        gated_activation (bool, optional): Defaults to True. If the gated activation is used.
        i_bn (bool, optional): Defaults to True. If the identity addition operation in residual layer is normalized.

    Returns:
        relu_out: A 4-D Tensor of shape [batch_size, Width, Height, out_channel]
    """

    in_shape = indata.get_shape().as_list()
    in_channel = in_shape[-1]
    with tf.variable_scope('identity_branch'):
        indata_cp = conv_layer(indata, ksize=[1, 1, in_channel, out_channel], padding='SAME', training=training,
                               name='identity', BN=i_bn, active=False)
    with tf.variable_scope('dilate_branch'):
        with tf.variable_scope('gate_branch'):
            gate_out = conv_layer(indata, ksize=[1, 2, in_channel, out_channel], padding='SAME', training=training,
                                  dilate=dilate, name='gate', BN=i_bn, active_function='sigmoid')
        with tf.variable_scope('filter_branch'):
            filter_out = conv_layer(indata, ksize=[1, 2, in_channel, out_channel], padding='SAME', training=training,
                                    dilate=dilate, name='filter', BN=i_bn, active_function='tanh')
        gated_out = gate_out * filter_out
        gated_cp = conv_layer(gated_out, ksize=[1, 1, out_channel, out_channel], padding='SAME', training=training,
                              name='identity', BN=i_bn, active=False, bias_term=False)
    with tf.variable_scope('plus'):
        relu_out = tf.nn.relu(indata_cp + gated_cp, name='final_relu')
    return relu_out


def getcnnfeature(signal, training, cnn_config):
    """Compute the CNN feature given the signal input.  

    Args:
        signal (Float): A 2D-Tensor of shape [batch_size,max_time]
        training (Boolean): A 0-D Boolean Tensor indicate if it's in training.      
        cnn_config(Dictionary): A dictionary indicate the configuration of CNN.
    Returns:
        cnn_fea: A 3D-Tensor of shape [batch_size, max_time, channel]
    """

    # TODO: Read the structure hyper parameters from Json file.
    signal_shape = signal.get_shape().as_list()
    batch_n = tf.shape(signal)[0]
    net = tf.reshape(signal, [batch_n, 1, signal_shape[1], 1])
    net_name = cnn_config['model']
    model_dict = {'dna_model1': DNA_model1, 
                  'rna_model1': RNA_model1,
                  'rna_model2' : RNA_model2,
                  'rna_model3' : RNA_model3,
                  'rna_test':rna_test,
                  'res_x': Res_x, 
                  'variant_wavnet': Variant_wavnet,
                  'incp_v2': incp_v2,
                  'custom': custom,
                  'gate_conv_net':gate_conv_net,
                  'gate_conv_net_low':gate_conv_net_low,
                  'gate_conv_net_high':gate_conv_net_high,
                  'dynamic_net':dynamic_net}
    if net_name == 'dynamic_net':
        net = model_dict[net_name](net,training,cnn_config)
    else:
        net = model_dict[net_name](net,training)
    feashape = net.get_shape().as_list()
    print("CNN output has the segment length %d, and %d channels"%(feashape[2],feashape[3]))
    net = tf.reshape(net, [batch_n, feashape[2],
                            feashape[3]], name='fea_rs')
    return net

def Res_x(net,training,layer_num = 10):
    #   Residual layer x 50
    for i in range(1,layer_num):
        with tf.variable_scope('res_layer'+str(i+1)):
            net = residual_layer(net,out_channel = 256,training = training,i_bn = True)
    return net

def DNA_model1(net,training):
    #   Residual Layer x 3 (DNA_default)
    with tf.variable_scope('res_layer1'):
        net = residual_layer(net, out_channel=256,
                              training=training, i_bn=True)
    with tf.variable_scope('res_layer2'):
        net = residual_layer(net, out_channel=256, training=training)
    with tf.variable_scope('res_layer3'):
        net = residual_layer(net, out_channel=256, training=training)
    return net

def RNA_model1(net,training):
    with tf.variable_scope('avg_pool1'):
        net = tf.nn.avg_pool(net, ksize = [1,3,1,1], strides = [1,1,2,1])
    with tf.variable_scope('res_layer1'):
        net = residual_layer(net, out_channel=256,
                              training=training, i_bn=True,strides = 2)
    with tf.variable_scope('res_layer2'):
        net = residual_layer(net, out_channel=256, training=training)
    with tf.variable_scope('res_layer3'):
        net = residual_layer(net, out_channel=256, training=training)
    return net

def dynamic_net(net, training, hp):
    """
    A function dynamically create a network based on the configuration.
    Args:
        net: A Tensor, the input network.
        training: If training.
        structure_hp: A string contain the structure information:
            tp: A list contain the type of each layers, must have same length of hu.
                  type can be one of the following: 
                      res: residual layer:
                      conv: convolutional layer
                      p_avg: average pooling layer
                      p_max: max pooling layer
            hu: A list contain the numbers of hidden unit for each layer.
            kw: A list contain the width of the kernals, must have same lenth of hu.
            st: A list contain the strides of the kernals, must have same length of hu.
            pd: A list contain the padding type. "VALID" or "SAME"
                      
    """
    hidden_num = hp['hu']
    kernals = hp['kw']
    strides = hp['st']
    layer_type = hp['tp']
    paddings = hp['pd']
    assert(len(hidden_num)==len(kernals)==len(strides)==len(layer_type)==len(paddings))
    for i,layer in enumerate(layer_type):
        with tf.variable_scope(layer+'_layer'+str(i)):
            if layer_type=='res':
                net = residual_layer(net,
                                     out_channel=hidden_num[i],
                                     training = training,
                                     k=kernals[i],
                                     strides = strides[i])
            elif layer_type=='conv':
                net = conv_layer(net,
                                 ksize = [1,kernals[i],1,hidden_num[i]],
                                 padding = paddings[i],
                                 training = training,
                                 name = 'conv')
            elif layer_type=='p_avg':
                net = tf.nn.avg_pool(net,
                                     ksize = [1,1,kernals[i],1],
                                     strides = [1,1,strides[i],1],
                                     padding = paddings[i])
            elif layer_type=='p_max':
                net = tf.nn.max_pool(net,
                                     ksize = [1,1,kernals[i],1],
                                     strides = [1,1,strides[i],1],
                                     padding = paddings[i])
    return net

def RNA_model2(net,training):
    with tf.variable_scope('res_layer1'):
        net = residual_layer(net, out_channel=256,
                              training=training, i_bn=True,strides = 2)
    with tf.variable_scope('res_layer2'):
        net = residual_layer(net, out_channel=256, training=training,strides = 2)
    with tf.variable_scope('res_layer3'):
        net = residual_layer(net, out_channel=256, training=training,strides = 2)
    return net

def RNA_model3(net,training):
    with tf.variable_scope('res_layer1'):
        net = residual_layer(net, out_channel=256,
                              training=training, k=13,i_bn=True,strides = 5)
    with tf.variable_scope('res_layer2'):
        net = residual_layer(net, out_channel=256, training=training)
    with tf.variable_scope('res_layer3'):
        net = residual_layer(net, out_channel=256, training=training)
    return net

def gate_conv_kernal(net,training,hp):
    """
    Kernal function of the gated convolution net
    https://arxiv.org/pdf/1609.03193.pdf
    https://arxiv.org/abs/1712.09444
    https://arxiv.org/pdf/1612.08083.pdf
    """
    fea_shape = net.get_shape().as_list()
    in_channel = fea_shape[-1]
    arch =  hp
    with tf.variable_scope('conv_1'):
        idx = 0
        net = conv_layer(net,
                         ksize=[1, arch['kw'][idx], in_channel, arch['hu'][idx]],
                         padding='SAME',
                         training=training,
                         name='conv1',
                         BN=True,
                         strides = arch['strides'][idx])
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv1'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv2'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv3'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv4'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv5'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv6'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1        
    with tf.variable_scope('gated_conv7'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('gated_conv8'):
        net = gated_conv_layer(net,kernal_width = arch['kw'][idx], out_channel = arch['hu'][idx], training = training)
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
        idx+=1
    with tf.variable_scope('fc_1'):
        net = conv_layer(net,ksize=[1,arch['kw'][idx], arch['hu'][idx-1], arch['hu'][idx]],padding='SAME',training = training,name = 'fc1')
        net = tf.nn.sigmoid(net)
        net = tf.nn.dropout(net,keep_prob = 1 - arch['dropout'][idx])
    return net

def gate_conv_net(net,training):
    arch = {'hu':[200]+ list(range(200,1601,200))+[1600],
            'kw':[17]+list(range(7,36,4))+[1],
            'dropout':[0]*10,
            'strides':[9]+ [1]* 9}
    net = gate_conv_kernal(net,training,hp = arch)
    return net
def gate_conv_net_low(net,training):
    arch = {'hu':[256,256,256,256,256,256,256,256,512,512],
            'kw':[9,3,3,3,3,3,3,3,21,1],
            'dropout':[0.25]+ list(np.asarray(range(8))*0.05 + 0.25) +[0.6],
            'strides':[3,1,1,1,1,1,1,1,1,1]}
    net = gate_conv_kernal(net,training,hp = arch)
    return net

def gate_conv_net_high(net,training):
    arch = {'hu':[200]+ list(range(200,1601,200))+[1600],
            'kw':[17]+list(range(7,36,4))+[1],
            'dropout':[0.25]+ list(np.asarray(range(8))*0.05 + 0.25) +[0.6],
            'strides':[9]+ [1]* 9}
    net = gate_conv_kernal(net,training,hp = arch)
    return net

def rna_test(net,training):
    with tf.variable_scope('res_layer1'):
        net = residual_layer(net,out_channel=256,training = training,i_bn = True)
    with tf.variable_scope('res_layer2'):
        net = residual_layer(net,out_channel = 256, training = training)
    with tf.variable_scope('res_layer3'):
        net = residual_layer(net,out_channel = 256, training = training)
    with tf.variable_scope('res_layer4'):
        net = residual_layer(net, out_channel=256, training=training)
    with tf.variable_scope('res_layer5'):
        net = residual_layer(net, out_channel=256, training=training)
    return net


            
def Variant_wavnet(net,training,res_layer = 1, dilate_layer = 7,dilate_repeat = 1):
    #   Dilate connection(Variant Wavenet) (res3_dilate7)
    with tf.variable_scope('res_layer1'):
        net = residual_layer(net,out_channel = 256,training = training,i_bn = True)
    for i in range(1,res_layer):
        with tf.variable_scope('res_layer'+str(i+1)):
            net = residual_layer(net,out_channel = 256,training = training)
    for block_idx in range(dilate_repeat):
        for i in range(dilate_layer):
           with tf.variable_scope('block'+str(block_idx+1)+'dilate_layer'+str(i+1)):
                net = wavenet_layer(net,out_channel = 256,training = training,dilate=2**i,i_bn=True)
    return net

def incp_v2(net,training):
    #Conv layer x 4
    with tf.variable_scope('conv_layer1'):
        net = conv_layer(net,ksize=[1,3,1,64],strides=[1,1,1,1],padding='SAME',training = training,name = 'conv')
    with tf.variable_scope('conv_layer2'):
        net = conv_layer(net,ksize=[1,3,64,128],strides=[1,1,1,1],padding='SAME',training= training,name = 'conv')
    with tf.variable_scope('conv_layer3'):
        net = conv_layer(net,ksize=[1,3,128,256],strides=[1,1,1,1],padding='SAME',training= training,name = 'conv')
    with tf.variable_scope('conv_layer4'):
        net = conv_layer(net,ksize=[1,5,256,256],strides=[1,1,1,1],padding='SAME',training= training,name = 'conv')

    #Inception layer x 9
    with tf.variable_scope('incp_layer1'):
        net = inception_layer(net,training)
    with tf.variable_scope('incp_layer2'):
        net = inception_layer(net,training)
    with tf.variable_scope('max_pool_1'):
        net = tf.nn.max_pool(net,ksize = [1,1,3,1],strides = [1,1,2,1],padding = 'SAME',name='mp_1x3_s2')
    with tf.variable_scope('incp_layer3'):
        net = inception_layer(net,training)
    with tf.variable_scope('incp_layer4'):
        net = inception_layer(net,training)
    with tf.variable_scope('incp_layer5'):
        net = inception_layer(net,training)
    with tf.variable_scope('incp_layer6'):
        net = inception_layer(net,training)
    with tf.variable_scope('incp_layer7'):
        net = inception_layer(net,training)
    with tf.variable_scope('max_pool_2'):
        net = tf.nn.max_pool(net,ksize = [1,1,3,1],strides = [1,1,2,1],padding = 'SAME',name='mp_1x3_s2')
    with tf.variable_scope('incp_layer8'):
        net = inception_layer(net,training)
    with tf.variable_scope('incp_layer9'):
        net = inception_layer(net,training)
    with tf.variable_scope('avg_pool_1'):
        net = tf.nn.avg_pool(net,ksize = [1,1,7,1],strides = [1,1,1,1],padding = 'SAME',name='ap_1x7_s1')
    return net

def custom(net,training):
    """define your customized CNN network here"""
    return net

def getcnnlogit(fea, outnum=5):
    """Get the logits from CNN feature.

    Args:
        fea (Float): A 3D-Tensor of shape [batch_size,max_time,channel]
        outnum (int, optional): Defaults to 5. Output class number, A,G,C,T,<ctc-blank>.

    Returns:
        A 3D-Tensor of shape [batch_size,max_time,outnum]
    """

    feashape = fea.get_shape().as_list()
    print(feashape)
    fea_len = feashape[-1]
    fea = tf.reshape(fea, [-1, fea_len])
    W = tf.get_variable("logit_weights", shape=[
                        fea_len, outnum], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("logit_bias", shape=[
                        outnum], initializer=tf.contrib.layers.xavier_initializer())
    return tf.reshape(tf.nn.bias_add(tf.matmul(fea, W), b, name='cnn_logits'), [feashape[0], feashape[1], outnum],
                      name='cnnlogits_rs')
