#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:32:32 2017

@author: haotianteng
"""
import tensorflow as tf
from distutils.dir_util import copy_tree
from chiron_input import read_raw_data_sets
from cnn import getcnnfeature
#from cnn import getcnnlogit
from rnn import rnn_layers
#from rnn import rnn_layers_one_direction

def save_model():
    copy_tree(FLAGS.home_dir+'/chiron/chiron',FLAGS.log_dir+FLAGS.model_name+'/model')
def inference(x,seq_length,training):
    cnn_feature = getcnnfeature(x,training = training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = FLAGS.sequence_len/feashape[1]
    logits = rnn_layers(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 )
#    logits = rnn_layers_one_direction(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 ) 
#    logits = getcnnlogit(cnn_feature)
    return logits,ratio

def loss(logits,seq_len,label):
    loss = tf.reduce_mean(tf.nn.ctc_loss(label,logits,seq_len,ctc_merge_repeated = True,time_major = False))
    """Note here ctc_loss will perform softmax, so no need to softmax the logits."""
    tf.summary.scalar('loss',loss)
    return loss

def train_step(loss):
    opt = tf.train.AdamOptimizer(FLAGS.step_rate).minimize(loss)
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

def train():
    training = tf.placeholder(tf.bool)
    x = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,FLAGS.sequence_len])
    seq_length = tf.placeholder(tf.int32, shape = [FLAGS.batch_size])
    y_indexs = tf.placeholder(tf.int64)
    y_values = tf.placeholder(tf.int32)
    y_shape = tf.placeholder(tf.int64)
    y = tf.SparseTensor(y_indexs,y_values,y_shape)
    logits,ratio = inference(x,seq_length,training)
    ctc_loss = loss(logits,seq_length,y)
    opt = train_step(ctc_loss)
    error = prediction(logits,seq_length,y)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    summary = tf.summary.merge_all()
    
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
    save_model()
    if FLAGS.retrain==False:
        sess.run(init)
        print("Model init finished, begin loading data. \n")
    else:
        saver.restore(sess,tf.train.latest_checkpoint(FLAGS.log_dir+FLAGS.model_name))
        print("Model loaded finished, begin loading data. \n")
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir+FLAGS.model_name+'/summary/', sess.graph)
    
    train_ds,valid_ds = read_raw_data_sets(FLAGS.data_dir,FLAGS.sequence_len,valid_reads_num = 10000,k_mer = FLAGS.k_mer)
    
    for i in range(FLAGS.max_steps):
        batch_x,seq_len,batch_y = train_ds.next_batch(FLAGS.batch_size)
        indxs,values,shape = batch_y
        feed_dict =  {x:batch_x,seq_length:seq_len/ratio,y_indexs:indxs,y_values:values,y_shape:shape,training:True}
        loss_val,_ = sess.run([ctc_loss,opt],feed_dict = feed_dict)
        if i%10 ==0:
            valid_x,valid_len,valid_y = valid_ds.next_batch(FLAGS.batch_size)
            indxs,values,shape = valid_y
            feed_dict = {x:valid_x,seq_length:valid_len/ratio,y_indexs:indxs,y_values:values,y_shape:shape,training:True}
            error_val = sess.run(error,feed_dict = feed_dict)
            print "Epoch %d, batch number %d, loss: %5.3f edit_distance: %5.3f"\
            %(train_ds.epochs_completed,train_ds.index_in_epoch,loss_val,error_val)
            saver.save(sess,FLAGS.log_dir+FLAGS.model_name+'/model.ckpt',i)
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
    print "Model %s saved."%(FLAGS.log_dir+FLAGS.model_name)
    print "Reads number %d"%(train_ds.reads_n)       
    saver.save(sess,FLAGS.log_dir+FLAGS.model_name+'/final.ckpt')
def run(args):
    global FLAGS
    FLAGS=args
    train()

if __name__ == "__main__":
    class Flags():
     def __init__(self):
        self.home_dir = "/shares/coin/haotian.teng/deepBNS/"
        self.data_dir = self.home_dir + 'data/train_set_all/raw/'
        self.log_dir = self.home_dir+'/chiron/log/'
        self.sequence_len = 300
        self.batch_size = 750
        self.step_rate = 1e-3 
        self.max_steps = 20000
        self.k_mer = 1
        self.model_name = 'crnn3+3_mix_hel'
        self.retrain = False
    run(Flags)
        
        
