#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:32:32 2017

@author: haotianteng
"""

import argparse
#from tensorflow.contrib.training.python.training import hparam
import tensorflow as tf
from distutils.dir_util import copy_tree
from chiron_input import read_raw_data_sets
from cnn import getcnnfeature
from cnn import getcnnlogit
from rnn import rnn_layers_one_direction
import time,os

def save_model(log_dir, model_name):
    copy_tree(os.path.dirname(os.path.abspath(__file__)),log_dir+model_name+'/model')
def inference(x,sequence_len,training):
    cnn_feature = getcnnfeature(x,training = training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = sequence_len/feashape[1]
    logits = getcnnlogit(cnn_feature)
    return logits,ratio

def loss(logits,seq_len,label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(label,logits,seq_len,ctc_merge_repeated = True,time_major = False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss',loss)
    return loss

def train_step(loss,step_rate, global_step = None):
    opt = tf.train.AdamOptimizer(step_rate).minimize(loss,global_step=global_step)
### Uncomment to use different optimizer
#    opt = tf.train.GradientDescentOptimizer(FLAGS.step_rate).minimize(loss)
#    opt = tf.train.RMSPropOptimizer(FLAGS.step_rate).minimize(loss)
#    opt = tf.train.MomentumOptimizer(FLAGS.step_rate,0.9).minimize(loss)
    return opt
def prediction(logits,seq_length,label,top_paths=1):
    """
    Args:
        logits:Input logits from a RNN.Shape = [batch_size,max_time,class_num]
        seq_length:sequence length of logits. Shape = [batch_size]
        label:Sparse tensor of label.
        top_paths:The number of top score path to choice from the decorder.
    """
    logits = tf.transpose(logits,perm = [1,0,2])
    predict = tf.nn.ctc_beam_search_decoder(logits,seq_length,merge_repeated = False,top_paths = top_paths)[0]
    edit_d = list()
    for i in range(top_paths):
        tmp_d = tf.edit_distance(tf.to_int32(predict[i]), label, normalize=True)
        edit_d.append(tmp_d)
    tf.stack(edit_d,axis=0)
    d_min = tf.reduce_min(edit_d,axis=0)
    error = tf.reduce_mean(d_min,axis = 0)
    tf.summary.scalar('Error_rate',error)
    return error

def train(hparam):
    training = tf.placeholder(tf.bool)
    global_step=tf.get_variable('global_step',trainable=False,shape=(),dtype = tf.int32,initializer = tf.zeros_initializer())
    x = tf.placeholder(tf.float32,shape = [hparam.batch_size,hparam.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape = [hparam.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs,y_values,y_shape)
    logits,ratio = inference(x,seq_length,training)
    ctc_loss = loss(logits,seq_length,y)
    opt = train_step(ctc_loss, hparam.step_rate, global_step = global_step)
    error = prediction(logits,seq_length,y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()
    
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
    save_model(hparam.log_dir, hparam.model_name)
    if hparam.retrain==False:
        sess.run(init)
        print("Model init finished, begin loading data. \n")
    else:
        saver.restore(sess,tf.train.latest_checkpoint(hparam.log_dir+hparam.model_name))
        print("Model loaded finished, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(hparam.log_dir+hparam.model_name+'/summary/', sess.graph)
    
    train_ds = read_raw_data_sets(hparam.data_dir,hparam.cache_dir,hparam.sequence_len,k_mer=hparam.kmer)
    start=time.time()
    for i in range(hparam.max_steps):
        batch_x,seq_len,batch_y = train_ds.next_batch(hparam.batch_size)
        indxs,values,shape = batch_y
        feed_dict =  {x:batch_x,seq_length:seq_len/ratio,y_indexs:indxs,y_values:values,y_shape:shape,training:True}
        loss_val,_ = sess.run([ctc_loss,opt],feed_dict = feed_dict)
        if i%10 ==0:
	    global_step_val = tf.train.global_step(sess,global_step)
            valid_x,valid_len,valid_y = train_ds.next_batch(hparam.batch_size)
            indxs,values,shape = valid_y
            feed_dict = {x:valid_x,seq_length:valid_len/ratio,y_indexs:indxs,y_values:values,y_shape:shape,training:True}
            error_val = sess.run(error,feed_dict = feed_dict)
	    end = time.time()
            print "Step %d/%d Epoch %d, batch number %d, loss: %5.3f edit_distance: %5.3f Elapsed Time/step: %5.3f"\
            %(i,hparam.max_steps,train_ds.epochs_completed,train_ds.index_in_epoch,loss_val,error_val,(end-start)/(i+1))
            saver.save(sess,hparam.log_dir+hparam.model_name+'/model.ckpt',global_step=global_step_val)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step = global_step_val)
            summary_writer.flush()
    global_step_val = tf.train.global_step(sess,global_step)
    print "Model %s saved."%(hparam.log_dir+hparam.model_name)
    print "Reads number %d"%(train_ds.reads_n)       
    saver.save(sess,hparam.log_dir+hparam.model_name+'/final.ckpt',global_step=global_step_val)

def run(hparam):
    train(hparam)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '--data-dir',
            help='Location containing training data',
            required=True
            )
    
    parser.add_argument(
            '--log-dir',
            help='Log dir location',
            required=True
            )
    parser.add_argument(
            '--sequence-len',
            help='Sequence length of nucleotides',
            default=300,
            type=int
            )
    parser.add_argument(
            '--batch-size',
            help='Training batch size',
            default=400,
            type=int
            )
    parser.add_argument(
            '--step-rate',
            help='Step rate',
            default=1e-3,
            type=float
            )
    parser.add_argument(
            '--max-steps',
            help='Max training steps',
            default=20000,
            type=int
            )
    parser.add_argument(
            '--kmer',
            help='K-mer length',
            default=1,
            type=int
            )
    parser.add_argument(
            '--model-name',
            help='Model name',
            required=True
            )
    parser.add_argument(
            '--retrain',
            help='Retrain the model',
            default=False,
            type=bool
            )
    #class Flags():
    # def __init__(self):
    #    self.data_dir = '/media/haotianteng/Linux_ex/Nanopore_data/Lambda_R9.4/raw/'
    #    self.cache_dir = '/media/haotianteng/Linux_ex/Nanopore_data/Lambda_R9.4/cache/train.hdf5'
    #    self.log_dir = '/media/haotianteng/Linux_ex/GVM_model'
    #    self.sequence_len = 300
    #    self.batch_size = 400
    #    self.step_rate = 1e-3 
    #    self.max_steps = 20000
    #    self.k_mer = 1
    #    self.model_name = 'res50'
    #    self.retrain =False
    #flags=Flags()
    #run(flags)
    args = parser.parse_args()
    run(hparam.HParams(**args.__dict__)) 
        
