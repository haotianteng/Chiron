#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:32:32 2017

@author: haotianteng
"""
import tensorflow as tf
from scipy import signal
import numpy as np
from ctcbns_input import read_raw_data_sets
from cnn import getcnnfeature
from rnn import rnn_layers
import matplotlib.pyplot as plt
import matplotlib
class Flags():
    def __init__(self):
        self.home_dir = "/home/haotianteng/UQ/deepBNS/"
        self.data_dir = self.home_dir + 'data/Lambda_R9.4/test/'
        self.log_dir = self.home_dir+'/ctcbns/log/'
        self.sequence_len = 150
        self.batch_size = 100
        self.step_rate = 1e-5 
        self.max_steps = 10000
        self.model_name = 'crnn5+5_res_test_bak'
FLAGS = Flags()

def inference(x,seq_length,training):
    cnn_feature = getcnnfeature(x,training = training)
    logits = rnn_layers(cnn_feature,seq_length,training)
    return cnn_feature,logits

def loss(logits,seq_len,label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(label,logits,seq_len,ctc_merge_repeated = False,time_major = False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss',loss)
    return loss

def train_step(loss):
    opt = tf.train.AdamOptimizer(FLAGS.step_rate).minimize(loss)
    return opt
def prediction(logits,seq_length,label):
    logits = tf.transpose(logits,perm = [1,0,2])
    """ctc_beam_search_decoder require input shape [max_time,batch_size,num_classes]"""
    predict = tf.to_int32(tf.nn.ctc_beam_search_decoder(logits,seq_length,merge_repeated = False)[0][0])
    error = tf.reduce_sum(tf.edit_distance(predict, label, normalize=False)) / tf.to_float(tf.size(label.values))
    tf.summary.scalar('Error_rate',error)
    return error

def morlet_cos_cwt(l,w):
    """
    Return the cwt friendly morlet function
    """
    return np.real(signal.morlet(l,w,s=2,complete=True))
def morlet_sin_cwt(l,w):
    """
    Return 
    """
    return np.imag(signal.morlet(l,w,s=2,complete=True))

def greedy_decoder(arg_logtis):
    """Decode the logits and output the logits position and corresponding logit"""
    pos=list()
    base = list()
    pre_base = 4
    pos_list = list()
    for x_indx,base_i in enumerate(arg_logits):
        if base_i==pre_base:
            pos_list.append(x_indx)
        else:
            if pre_base!=4:
                pos.append(np.mean(pos_list))
                base.append(pre_base)
            pos_list=[]
            pos_list.append(x_indx)
            
        pre_base=base_i
    return pos,base
                
            
    
    
"""Copy the train function here"""
train_ds,valid_ds = read_raw_data_sets(FLAGS.data_dir,FLAGS.sequence_len,valid_reads_num = 100)
with tf.device('/gpu:0'):
    training = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape = [FLAGS.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs,y_values,y_shape)
    cnn_feature,logits = inference(x,seq_length,training)
    ctc_loss = loss(logits,seq_length,y)
    opt = train_step(ctc_loss)
    error = prediction(logits,seq_length,y)
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
summary_writer = tf.summary.FileWriter(FLAGS.log_dir+FLAGS.model_name+'/summary/', sess.graph)
#sess.run(init)
#saver.restore(sess,FLAGS.log_dir+FLAGS.model_name+'/model.ckpt-9070')
saver.restore(sess,tf.train.latest_checkpoint(FLAGS.log_dir+FLAGS.model_name))
batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size,shuffle = False)
indxs,values,shape = batch_y
feed_dict = {x:batch_x,seq_length:seq_len,y_indexs:indxs,y_values:values,y_shape:shape,training:False}
loss_val = sess.run([ctc_loss],feed_dict = feed_dict)
cnn_feature_val = sess.run(cnn_feature,feed_dict=feed_dict)
#valid_x,valid_len,valid_y = valid_ds.next_batch(FLAGS.batch_size)
#feed_dict = {x:batch_x,seq_length:seq_len,y_indexs:indxs,y_values:values,y_shape:shape,training:False}
error_val = sess.run(error,feed_dict = feed_dict)
""""""

"""Conduct test"""
#batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size)
#feed_dict = {x:batch_x,seq_length:seq_len,y_indexs:indxs,y_values:values,y_shape:shape,training:True}
predict = tf.nn.ctc_beam_search_decoder(tf.transpose(logits,perm=[1,0,2]),seq_length,merge_repeated = False,top_paths = 5)
predict_val = sess.run(predict,feed_dict = feed_dict)
predict_val_top5 = predict_val[0]
index_val = sess.run(y_indexs,feed_dict = feed_dict)
y_val_eval = sess.run(y_values,feed_dict = feed_dict)
index_val_bat = index_val[:,0]
predict_read = list()
true_read = list()
for i in range(len(predict_val_top5)):
    predict_val = predict_val_top5[i]
    unique,len_counts = np.unique(index_val_bat,return_counts = True)
    unique,pre_counts = np.unique(predict_val.indices[:,0],return_counts = True)
    pos_predict = 0
    pos_true = 0
    predict_read_temp = list()
    true_read_temp = list()
    for indx,counts in enumerate(len_counts):
        predict_read_temp.append(predict_val.values[pos_predict:pos_predict+pre_counts[indx]])
        pos_predict +=pre_counts[indx]
        true_read_temp.append(y_val_eval[pos_true:pos_true+counts])
        pos_true+=counts
    true_read.append(true_read_temp)
    predict_read.append(predict_read_temp)
""""""

"""logits plot"""
matplotlib.rcParams.update({'font.size': 28})
inspect_indx = 1
print(predict_read[0][inspect_indx].tolist())
print(true_read[0][inspect_indx].tolist())
logits_val = sess.run(tf.nn.softmax(logits),feed_dict=feed_dict)
logits_val = logits_val[inspect_indx]
cnn_feature=np.matrix.transpose(cnn_feature_val[10])[:100]
A_logits = logits_val[:,0]
C_logits = logits_val[:,1]
G_logits = logits_val[:,2]
T_logits = logits_val[:,3]
b_logits = logits_val[:,4]
arg_logits = np.argmax(logits_val,axis=1)

x_val = sess.run(x,feed_dict = feed_dict)
x_val = x_val[inspect_indx]
widths = np.arange(1, 7, 0.5)
#cwtmatr_cos = signal.cwt(x_val, morlet_cos_cwt, widths)
cwtmatr = signal.cwt(x_val, signal.ricker, widths)
#cwtmatr = np.concatenate((cwtmatr_cos,cwtmatr_sin),axis=0)
plt.plot(x_val)
f,axarr = plt.subplots(2,sharex=True)
axarr[0].plot(x_val,color='black')
[tk.set_visible(True) for tk in axarr[0].get_xticklabels()]
#axarr[1].imshow(cnn_feature)


C_h=axarr[1].plot(C_logits,color = 'g',label='C')
G_h=axarr[1].plot(G_logits,color = 'b',label='G')
T_h=axarr[1].plot(T_logits,color = '#ffc800',label='T')
A_h=axarr[1].plot(A_logits,color = 'r',label='A')
b_h=axarr[1].plot(b_logits,color = 'black',alpha=0.5,linestyle='dashed',label='B')
L=axarr[1].legend([A_h[0],C_h[0],G_h[0],T_h[0],b_h[0]],["$A$","$C$","$G$","$T$","$b$"],bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.,handletextpad = 0)
[tk.set_visible(False) for tk in axarr[1].get_xticklabels()]
axarr[1].set_ylim(ymin=0)
for t in L.get_texts():
    t.set_ha('left')
    t.set_position((6,0))
#plt.setp(L.texts, family='Times New Roman')
#axarr[1].set_title('Base prediction')
#axarr[1].set_title('CNN feature output')
#axarr[0].set_title('signal')

axarr[0].set_ylabel('relative strength')
axarr[1].set_ylabel('label probability')
#axarr[1].set_ylabel('channel index')
#axarr[2].imshow(cwtmatr, cmap='PRGn', aspect='auto',vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
base_dict=['A','C','G','T']
pos,base = greedy_decoder(arg_logits)
pos[4]=pos[4]+0.5
for indx,base_i in enumerate(base):
    axarr[1].text(pos[indx]-1,-0.06,base_dict[base_i],fontsize=15)
#    axarr[0].axvline(x=pos[indx],ymin=-1.2,ymax=1,color='black',linestyle='dotted',zorder=0,clip_on=False)
#    axarr[1].axvline(x=pos[indx],ymin=0.03,ymax=1,color='black',linestyle='dotted',zorder=0,clip_on=False)