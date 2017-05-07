#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:59:15 2017

@author: haotianteng
"""
import argparse,os,time
import numpy as np
import tensorflow as tf
from ctcbns_input import read_data_for_eval
from ctcbns.utils.easy_assembler import simple_assembly
from cnn import getcnnfeature
#from cnn import getcnnlogit
from rnn import rnn_layers

parser = argparse.ArgumentParser(description='Basecall a signal file')
parser.add_argument('-i','--input', help="File path or Folder path to the signal file.")
parser.add_argument('-o','--output', help = "Output Folder name")
parser.add_argument('-m','--model', default = '../log/crnn5+5_resnet',help = "model folder")
parser.add_argument('-s','--start',type=int,default = 0,help = "Start index of the signal file.")
parser.add_argument('-b','--batch_size',type = int,default = 100,help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
parser.add_argument('-l','--segment_len',type = int,default = 200, help="Segment length to be divided into.")
parser.add_argument('-j','--jump',type = int,default = 20,help = "Step size for segment")
FLAGS = parser.parse_args()

def inference(x,seq_length,training):
    cnn_feature = getcnnfeature(x,training = training)
    feashape = cnn_feature.get_shape().as_list()
    ratio = FLAGS.segment_len/feashape[1]
    logits = rnn_layers(cnn_feature,seq_length/ratio,training,class_n = 5 )
#    logits = rnn_layers_one_direction(cnn_feature,seq_length/ratio,training,class_n = 4**FLAGS.k_mer+1 ) 
#    logits = getcnnlogit(cnn_feature)
    return logits,ratio

def sparse2dense(predict_val):
    predict_val_top5 = predict_val[0]
    predict_read = list()
    for i in range(len(predict_val_top5)):
        predict_val = predict_val_top5[i]
        unique,pre_counts = np.unique(predict_val.indices[:,0],return_counts = True)
        pos_predict = 0
        predict_read_temp = list()
        for indx,counts in enumerate(pre_counts):
            predict_read_temp.append(predict_val.values[pos_predict:pos_predict+pre_counts[indx]])
            pos_predict +=pre_counts[indx]
        predict_read.append(predict_read_temp)
    return predict_read
def index2base(read):
    base = ['A','C','G','T']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread


x = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,FLAGS.segment_len])
seq_length = tf.placeholder(tf.int32, shape = [FLAGS.batch_size])
training = tf.placeholder(tf.bool)
logits,_ = inference(x,seq_length,training = training)
predict = tf.nn.ctc_beam_search_decoder(tf.transpose(logits,perm=[1,0,2]),seq_length,merge_repeated = False,top_paths = 5)
    
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
saver = tf.train.Saver()
saver.restore(sess,tf.train.latest_checkpoint(FLAGS.model))
if os.path.isdir(FLAGS.input):
    file_list = os.listdir(FLAGS.input)
    file_dir = FLAGS.input
else:
    file_list = [os.path.basename(FLAGS.input)]
    file_dir = os.path.abspath(os.path.join(FLAGS.input,os.path.pardir))
for name in file_list:
    start_time = time.time()
    if not name.endswith('.signal'):
        continue
    file_pre = os.path.splitext(name)[0]
    input_path = os.path.join(file_dir,name)
    eval_data = read_data_for_eval(input_path,FLAGS.start,seg_length = FLAGS.segment_len,step = FLAGS.jump)
    reads_n = eval_data.reads_n
    reads = list()
    for i in range(0,reads_n,FLAGS.batch_size):
        batch_x,seq_len,_ = eval_data.next_batch(FLAGS.batch_size,shuffle = False)
        feed_dict = {x:batch_x,seq_length:seq_len,training:False}
        predict_val = sess.run(predict,feed_dict = feed_dict)
        predict_read = sparse2dense(predict_val)
        predict_read = predict_read[0]#Top 1 path
        if i+FLAGS.batch_size>reads_n:
            predict_read = predict_read[:reads_n-i]
        reads+=predict_read
    print("Segment reads base calling finished, begin to assembly. %5.2f seconds"%(time.time()-start_time))
    bpreads = [index2base(read) for read in reads]
    concensus = simple_assembly(bpreads)
    c_bpread = index2base(np.argmax(concensus,axis = 0))
    print("Assembly finished, begin output. %5.2f seconds"%(time.time()-start_time))
    result_folder = os.path.join(FLAGS.output,'result')
    seg_folder = os.path.join(FLAGS.output,'segments')
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)
    if not os.path.exists(seg_folder):
        os.makedirs(seg_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    path_con = os.path.join(result_folder,file_pre+'.fasta')
    path_reads = os.path.join(seg_folder,file_pre+'.fasta')
    out_f = open(path_reads,'w+')
    out_con = open(path_con,'w+')
    for indx,read in enumerate(bpreads):
        out_f.write(">sequence"+str(indx)+'\n')
        out_f.write(read+'\n')
    out_con.write(">sequence\n"+c_bpread)
    basecall_time = time.time()-start_time
    total_len = len(c_bpread)                        
    print("Basecalled %s in %5.2f seconds,%5.2f bp/s"%(name,basecall_time,total_len/basecall_time))    
