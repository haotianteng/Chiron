#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 02:48:26 2017

@author: haotianteng
"""

import tensorflow as tf
from tensorflow.contrib.layers import batch_norm

def conv_layer(indata,ksize,padding,training,name,dilate = 1,strides=[1,1,1,1],bias_term = False,active = True,BN= True):
    """A standard convlotional layer"""
    with tf.variable_scope(name):
        W = tf.get_variable("weights", dtype = tf.float32, shape=ksize,initializer=tf.contrib.layers.xavier_initializer())
        if bias_term:
            b = tf.get_variable("bias", dtype=tf.float32,shape=[ksize[-1]])
        if dilate>1:
            if bias_term:
                conv_out = b + tf.nn.atrous_conv2d(indata,W,rate = dilate,padding=padding,name=name)
            else:
                conv_out = tf.nn.atrous_conv2d(indata,W,rate = dilate,padding=padding,name=name)
        else:
            if bias_term:
                conv_out = b + tf.nn.conv2d(indata,W,strides = strides,padding = padding,name = name)
            else:
                conv_out = tf.nn.conv2d(indata,W,strides = strides,padding = padding,name = name)
    if BN:
        with tf.variable_scope(name+'_bn') as scope:
#            conv_out = batchnorm(conv_out,scope=scope,training = training)
            conv_out = simple_global_bn(conv_out,name = name+'_bn')
    if active:
        with tf.variable_scope(name+'_relu'):
            conv_out = tf.nn.relu(conv_out,name='relu')
    return conv_out
def batchnorm(inp,scope,training,decay = 0.99,epsilon = 1e-5):
    with tf.variable_scope(scope):
        size = inp.get_shape().as_list()[-1]
        scale = tf.get_variable('scale', shape = [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', shape = [size])

        pop_mean = tf.get_variable('pop_mean', shape = [size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', shape = [size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean,batch_var = tf.nn.moments(inp,[0,1,2])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(inp, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(inp, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)
def simple_global_bn(inp,name):
    ksize = inp.get_shape().as_list()
    ksize = [ksize[-1]]
    mean,variance = tf.nn.moments(inp,[0,1,2],name=name+'_moments')
    scale = tf.get_variable(name+"_scale", shape=ksize)#,initializer=tf.contrib.layers.variance_scaling_initializer())
    offset = tf.get_variable(name+"_offset", shape=ksize)#,initializer=tf.contrib.layers.variance_scaling_initializer())
    return tf.nn.batch_normalization(inp,mean=mean,variance=variance,scale=scale,offset=offset,variance_epsilon=1e-5)
def inception_layer(indata,training,times=16):
    """Inception module with dilate conv layer from http://arxiv.org/abs/1512.00567"""
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1_AvgPooling'):
        avg_pool = tf.nn.avg_pool(indata,ksize = (1,1,3,1),strides = (1,1,1,1),padding = 'SAME',name = 'avg_pool0a1x3')
        conv1a = conv_layer(avg_pool,ksize=[1,1,in_channel,times*3],padding = 'SAME',training = training,name = 'conv1a_1x1')
    with tf.variable_scope('branch2_1x1'):
        conv0b = conv_layer(indata,ksize=[1,1,in_channel,times*3],padding = 'SAME',training = training,name = 'conv0b_1x1')
    with tf.variable_scope('branch3_1x3'):
        conv0c = conv_layer(indata,ksize=[1,1,in_channel,times*2],padding = 'SAME',training = training,name = 'conv0c_1x1')
        conv1c = conv_layer(conv0c,ksize=[1,3,times*2,times*3],padding = 'SAME',training = training,name = 'conv1c_1x3')
    with tf.variable_scope('branch4_1x5'):
        conv0d = conv_layer(indata,ksize=[1,1,in_channel,times*2],padding = 'SAME',training = training,name = 'conv0d_1x1')
        conv1d = conv_layer(conv0d,ksize=[1,5,times*2,times*3],padding = 'SAME',training = training,name = 'conv1d_1x5')
    with tf.variable_scope('branch5_1x3_dilate_2'):
        conv0e = conv_layer(indata,ksize=[1,1,in_channel,times*2],padding = 'SAME',training = training,name = 'conv0e_1x1')
        conv1e = conv_layer(conv0e,ksize=[1,3,times*2,times*3],padding = 'SAME',training = training,name = 'conv1e_1x3_d2',dilate = 2)
    with tf.variable_scope('branch6_1x3_dilate_3'):
        conv0f = conv_layer(indata,ksize=[1,1,in_channel,times*2],padding = 'SAME',training = training,name = 'conv0f_1x1')
        conv1f = conv_layer(conv0f,ksize=[1,3,times*2,times*3],padding = 'SAME',training = training,name = 'conv1f_1x3_d3',dilate = 3)
    return(tf.concat([conv1a,conv0b,conv1c,conv1d,conv1e,conv1f],axis = -1,name = 'concat'))
def residual_layer(indata,out_channel,training,i_bn = False):
    fea_shape = indata.get_shape().as_list()
    in_channel = fea_shape[-1]
    with tf.variable_scope('branch1'):
        indata_cp = conv_layer(indata,ksize = [1,1,in_channel,out_channel],padding = 'SAME',training = training,name = 'conv1',BN = i_bn,active = False)
    with tf.variable_scope('branch2'):
        conv_out1 = conv_layer(indata,ksize = [1,1,in_channel,out_channel],padding = 'SAME',training = training,name = 'conv2a',bias_term = False)
        conv_out2 = conv_layer(conv_out1,ksize = [1,3,out_channel,out_channel],padding = 'SAME',training=training,name = 'conv2b',bias_term = False)
        conv_out3 = conv_layer(conv_out2,ksize = [1,1,out_channel,out_channel],padding = 'SAME',training=training,name = 'conv2c',bias_term = False,active = False)
    with tf.variable_scope('plus'):
        relu_out = tf.nn.relu(indata_cp+conv_out3,name = 'final_relu')
    return relu_out
def getcnnfeature(signal,training):
    signal_shape = signal.get_shape().as_list()
    signal = tf.reshape(signal,[signal_shape[0],1,signal_shape[1],1])
    print(signal.get_shape())
#    #Conv layer x 4
#    with tf.variable_scope('conv_layer1'):
#        conv1 = conv_layer(signal,ksize=[1,3,1,256],strides=[1,1,1,1],padding='SAME',training = training,name = 'conv')
#    with tf.variable_scope('conv_layer2'):    
#        conv2 = conv_layer(conv1,ksize=[1,5,256,256],strides=[1,1,1,1],padding='SAME',training= training,name = 'conv')
#    with tf.variable_scope('conv_layer3'):    
#        conv3 = conv_layer(conv2,ksize=[1,5,256,256],strides=[1,1,1,1],padding='SAME',training= training,name = 'conv')
#    with tf.variable_scope('conv_layer4'):    
#        conv4 = conv_layer(conv3,ksize=[1,5,256,256],strides=[1,1,1,1],padding='SAME',training= training,name = 'conv')
#    with tf.variable_scope('conv_layer5'):    
#        conv5 = conv_layer(conv4,ksize=[1,5,256,256],strides=[1,1,2,1],padding='SAME',training= training,name = 'conv')

#    Inception layer x 9
#    with tf.variable_scope('incp_layer1'):    
#        incp5 = inception_layer(conv4,training)
#    with tf.variable_scope('incp_layer2'):
#        incp6 = inception_layer(incp5,training)
#        
#    with tf.variable_scope('max_pool_1'):
#        max_pool1 = tf.nn.max_pool(incp6,ksize = [1,1,3,1],strides = [1,1,2,1],padding = 'SAME',name='mp_1x3_s2')
#        
#    with tf.variable_scope('incp_layer3'):
#        incp7 = inception_layer(max_pool1,training)
#    with tf.variable_scope('incp_layer4'):
#        incp8 = inception_layer(incp7,training)
#    with tf.variable_scope('incp_layer5'):
#        incp9 = inception_layer(incp8,training)
#    with tf.variable_scope('incp_layer6'):
#        incp10 = inception_layer(incp9,training)
#    with tf.variable_scope('incp_layer7'):
#        incp11 = inception_layer(incp10,training)
#        
#    with tf.variable_scope('max_pool_2'):
#        max_pool2 = tf.nn.max_pool(incp11,ksize = [1,1,3,1],strides = [1,1,2,1],padding = 'SAME',name='mp_1x3_s2')
#    
#    with tf.variable_scope('incp_layer8'):
#        incp12 = inception_layer(max_pool2,training)
#    with tf.variable_scope('incp_layer9'):
#        incp13 = inception_layer(incp12,training)
#    
#    with tf.variable_scope('avg_pool_1'):
#        avg_pool1 = tf.nn.avg_pool(incp13,ksize = [1,1,7,1],strides = [1,1,1,1],padding = 'SAME',name='ap_1x7_s1')

#   Residual Layer x 5
    with tf.variable_scope('res_layer1'):
        res1 = residual_layer(signal,out_channel = 256,training = training,i_bn = True)
    with tf.variable_scope('res_layer2'):
        res2 = residual_layer(res1,out_channel = 256,training = training)
    with tf.variable_scope('res_layer3'):
        res3 = residual_layer(res2,out_channel = 256,training = training)
    with tf.variable_scope('res_layer4'):
        res4 = residual_layer(res3,out_channel = 512,training = training)
    with tf.variable_scope('res_layer5'):
        res5 = residual_layer(res4,out_channel = 512,training = training)
        
    feashape = res5.get_shape().as_list()
    fea = tf.reshape(res5,[feashape[0],feashape[2],feashape[3]],name = 'fea_rs')
    return fea

def getcnnlogit(fea,outnum=5):
    feashape = fea.get_shape().as_list()
    print feashape
    fea_len = feashape[-1]
    fea = tf.reshape(fea,[-1,fea_len])
    W = tf.get_variable("logit_weights", shape=[fea_len,outnum],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("logit_bias", shape=[outnum],initializer=tf.contrib.layers.xavier_initializer())
    return tf.reshape(tf.nn.bias_add(tf.matmul(fea,W),b,name = 'cnn_logits'),[feashape[0],feashape[1],outnum],name = 'cnnlogits_rs')
