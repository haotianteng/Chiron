#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 17:12:16 2017

@author: haotianteng
"""
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.framework import ops    
from tensorflow.python.ops import variable_scope as vs
import numpy as np
def attention_loss(inputs,seq_len,label_dense,label_len,output_size=5,cell_size=200):
    """
    Args:
    inputs: output from a encoder, [batch_size,max_seq_len,hidden_num].
    seq_len:Length of the sequence.
    label_dense:A dense matrix of label batch[batch_size,max_label_len].
    label_len:[batch_size] label length, the <end> symbol is included.
    seq_len,label_len: 1D tensor with [batch_size]
    """
    cell=GRUCell(cell_size)
    size_list=inputs.get_shape().as_list()
    max_seq_len=size_list[1]
    logits,attention_list=attention_decoder(inputs,seq_len,cell,max_label_len=max_seq_len,label=label_dense)
    label_mask=tf.sequence_mask(label_len,maxlen=max_seq_len,dtype=tf.float32,name='label_mask')
#    label_mask = tf.get_variable(name = 'test_mask',shape=[batch_size,max_seq_len])
    loss = tf.contrib.seq2seq.sequence_loss(logits,label_dense,label_mask,average_across_timesteps=False)
    return loss,logits,attention_list
def attention_decoder(inputs,seq_length,cell,label=None,var_scope="attention_decoder",max_label_len=None,attention_size=200,initial_state=None,initial_attention=None,dtype=tf.float32,channel_num=20,kernal_width=5,k_mer=1,for_eval = False):
    """Args:
    inputs:3D Tensor [batch_size,max_seq_length,hidden_num] from the encoder output
    seq_length:List, size of [batch_size], the real length of sequence
    cell:decoder cell, state [batch_size, state_size],output [batch_size,output_base+1]
    attention_size:Internal attention vector size,only compute the highest attention_size.
    initial_state:Tensor of same size of cell. Initial state size of the decoder RNN cell, if None, the cell will be initialized as zero.
    initial_attention:Tensor with size [batch_size,max_seq_length].Initial attention state, if None, initial with zeros.
    dtype:data type, if None use float32.
    channel_num:Covolution kernal output channel number of F, see https://arxiv.org/abs/1506.07503 equation 8.
    
    Return:
    output logits [batch_size,seq_len,logits_num=5]
    """
    size_list=inputs.get_shape().as_list()
    batch_size=size_list[0]
    max_seq_len=size_list[1]
    if max_label_len is None:
        """If there is no label length input, usually when in evaluation."""
        max_label_len=int(max_seq_len)
    hidden_num=size_list[2]
    cell_size=cell.state_size
    if not isinstance(cell_size,int):   
        cell_size=cell_size[0]
    class_n=4**k_mer+1
    with tf.variable_scope(var_scope,initializer = tf.random_normal_initializer(stddev = 0.01)) as scope:
        if initial_state is None:
            state=cell.zero_state(batch_size=batch_size,dtype=dtype)
        else:
            state = initial_state
        if initial_attention is None:
            attention=tf.zeros(name='attention',shape=[batch_size,max_seq_len,1],dtype=dtype)
        else:
            attention_size=attention.get_shape().as_list()
            if len(attention_size)==2:
                attention=tf.reshape(attention,shape=attention_size+[1])
            attention_size=attention.get_shape().as_list()
            assert attention_size==[batch_size,max_seq_len,1]
            attention=tf.Variable(initial_attention,name='attention')
        F = tf.get_variable("F", dtype = tf.float32, shape=(1,kernal_width,1,channel_num),initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        W = tf.get_variable(name='W',shape=[cell_size,attention_size])
        V = tf.get_variable(name='V',shape=[hidden_num,attention_size])
        U = tf.get_variable(name='U',shape=[channel_num,attention_size])
        b = tf.get_variable(name='bias',shape=[batch_size,attention_size],initializer=tf.zeros_initializer())
        w = tf.get_variable(name='w',shape=[attention_size,1])
        seq_mask=tf.sequence_mask(seq_length,maxlen=max_seq_len,dtype=dtype,name='seq_mask')
        seq_mask=tf.reshape(seq_mask,[batch_size,max_seq_len,1])
        
        def _attention_sum(inputs,attention,sharpening=1):
            ###Need further imporvement, including the sharpenning and smoothing technique
            """
            Return:
                a: [batch_size,max_seq_len,1]
                sum:[batch_size,hidden_num]
            """
            e = attention*seq_mask
            a = tf.nn.softmax(sharpening*e,dim=1)
            return a,tf.reduce_sum(inputs*a,axis=1)
        
        def _emition(state,h_sum,hidden_size=100):
            #https://arxiv.org/abs/1409.0473
            """Args:
                state: [batch_size,cell_size]
                h_sum: [batch_size,hidden_num]
               Return:
                y:[batch_size,class_n]
            """
            uo=tf.get_variable(name='u_o',shape=[cell_size,hidden_size])
            co=tf.get_variable(name='c_o',shape=[hidden_num,hidden_size])
            wo=tf.get_variable(name="w_o",shape=[hidden_size/2,class_n])
            t=tf.matmul(state,uo)+tf.matmul(h_sum,co)
            t = tf.reshape(t,shape=[batch_size,hidden_size,1])
            t_max=tf.nn.pool(input=t,window_shape=[2],pooling_type='MAX',strides=[2],padding='VALID')
            t_max=tf.reshape(t_max,shape=[batch_size,hidden_size/2])
            y=tf.matmul(t_max,wo)
            return y
        
        def _transition(attention_sum,output):
            #https://arxiv.org/abs/1409.0473
            """Args:
                attention_sum:[batch_size,hidden_num]
                pre_state:[batch_size,cell_size]
                output:[batch_size,output_size]
               Return:
                next_input:[batch_size,cell_size]
                
            """
            C = tf.get_variable(name='C',shape=[hidden_num,cell_size])
            E = tf.get_variable(name='E',shape=[class_n,cell_size])
            next_input=tf.matmul(attention_sum,C)+tf.matmul(output,E)
            return next_input
            
        """ TODO using tf.while_loop instead of python for loop
        a,mul_sum=_attention_sum(inputs,attention)
        output,state = cell(mul_sum,state)
        with vs.variable_scope("output_rnn") as varscope:
            output,state = cell(mul_sum,state)
            output_list.append(output)
            def _loop_cond(i,a,state,attention,output):
                return tf.less(i,max_label_len)
            def _loop_function(i,a,state,attention,output):
                varscope.reuse_variables()
                a = tf.reshape(a,shape=[batch_size,max_seq_len,1,1])
                conv=tf.nn.conv2d(a,F,strides=[1,1,1,1],padding='SAME')
                conv=tf.reshape(conv,shape=[batch_size,max_seq_len,channel_num])
                
                content = tf.matmul(state,W,name='WS_matmul')+tf.stack([b]*batch_size,axis=0)
                loc_cont=tf.stack([content]*max_seq_len,axis=1)+matmul(inputs,V)+matmul(conv,U)
                new_attention = matmul(tf.tanh(loc_cont),w)
                a,mul_sum=_attention_sum(inputs,attention)
                (next_output,new_state) = cell(mul_sum,state)
                output_list.append(output)
                return tf.add(i,1),a,new_state,new_attention,next_output
            #Begin the loop
            i = tf.constant(0)
            result = tf.while_loop(_loop_cond, _loop_function, (i,a,state,attention,output))[4]
        """
        output_list=list()
        attention_list=list()
        a,mul_sum=_attention_sum(inputs,attention) #a_0
        for index in range(max_label_len):
            
            #Calcualte new attention
            a = tf.reshape(a,shape=[batch_size,max_seq_len,1,1])
            conv=tf.nn.conv2d(a,F,strides=[1,1,1,1],padding='SAME')
            conv=tf.reshape(conv,shape=[batch_size,max_seq_len,channel_num])
            
            content = tf.matmul(state,W,name='WS_matmul')+b
            content_flat = tf.reshape(content,[-1])
            location=matmul(inputs,V)+matmul(conv,U)#[batch_size,max_seq_len,attention_size]
            location_t=tf.transpose(location,perm=[1,0,2])#[max_seq_len,batch_size,attention_size]
            location_flat = tf.reshape(location_t,shape=[max_seq_len,batch_size*attention_size])
            loc_cont_flat = location_flat+content_flat
            loc_cont_tr = tf.reshape(loc_cont_flat,shape=[max_seq_len,batch_size,attention_size])
            loc_cont = tf.transpose(loc_cont_tr,perm=[1,0,2])
            attention = matmul(tf.tanh(loc_cont),w)
            a,mul_sum=_attention_sum(inputs,attention)
            attention_list.append(a)
            
            #Emition
            with vs.variable_scope("emition") as varscope:
                if index>0:
                    varscope.reuse_variables()
                y=_emition(state,mul_sum)
                    
            #Calculate output
            with vs.variable_scope("transition") as varscope:
                if index>0:
                    varscope.reuse_variables()
                if for_eval:
                    cell_input=_transition(mul_sum,y)
                else:
                    y_hat = tf.one_hot(label[:,index],depth=5,on_value=1.0,off_value=0.0,axis=-1)
                    cell_input=_transition(mul_sum,y_hat)
            with vs.variable_scope("rnn_output",initializer=tf.orthogonal_initializer()) as varscope:
                if index>0:
                    varscope.reuse_variables()
                output,state = cell(cell_input,state)
                
            output_list.append(y)
    return tf.stack(output_list,axis=1),attention_list
def matmul(a,b):
    """
    A expanded method for tf.matmul
    a is a tensor shapped [...,n]
    b is matrix shapped [n,m]
    Output is [...,m]
    """
    a_shape=a.get_shape().as_list()
    a_last=a_shape[-1]
    except_last=a_shape[:-1]
    b_last=b.get_shape().as_list()[-1]
    c = tf.matmul(tf.reshape(a,shape=[-1,a_last]),b)
    return(tf.reshape(c,shape=except_last+[b_last]))