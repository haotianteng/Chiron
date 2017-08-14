#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:59:15 2017

@author: haotianteng
"""
import argparse,os,time
import numpy as np
import tensorflow as tf
from chiron_input import read_data_for_eval
from utils.easy_assembler import simple_assembly
from cnn import getcnnfeature
#from cnn import getcnnlogit
from rnn import rnn_layers
from utils.unix_time import unix_time

parser = argparse.ArgumentParser(description='Basecall a signal file')
parser.add_argument('-i','--input', help="File path or Folder path to the signal file.")
parser.add_argument('-o','--output', help = "Output Folder name")
parser.add_argument('-m','--model', default = './model/DNA_default',help = "model folder")
parser.add_argument('-s','--start',type=int,default = 0,help = "Start index of the signal file.")
parser.add_argument('-b','--batch_size',type = int,default = 1100,help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
parser.add_argument('-l','--segment_len',type = int,default = 300, help="Segment length to be divided into.")
parser.add_argument('-j','--jump',type = int,default = 30,help = "Step size for segment")
parser.add_argument('-t','--threads',type = int,default = 0,help = "Threads number")
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
    bpread = [base[x] for x in read]6
    bpread = ''.join(x for x in bpread)
    return bpread

def evaluation():
    x = tf.placeholder(tf.float32,shape = [FLAGS.batch_size,FLAGS.segment_len])
    seq_length = tf.placeholder(tf.int32, shape = [FLAGS.batch_size])
    training = tf.placeholder(tf.bool)
    logits,_ = inference(x,seq_length,training = training)
    predict = tf.nn.ctc_greedy_decoder(tf.transpose(logits,perm=[1,0,2]),seq_length,merge_repeated = True)
    config=tf.ConfigProto(allow_soft_placement=True,intra_op_parallelism_threads=FLAGS.threads,inter_op_parallelism_threads=FLAGS.threads)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as sess:
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
         reading_time=time.time()-start_time
         reads = list()
         for i in range(0,reads_n,FLAGS.batch_size):
             batch_x,seq_len,_ = eval_data.next_batch(FLAGS.batch_size,shuffle = False)
             batch_x=np.pad(batch_x,((0,FLAGS.batch_size-len(batch_x)),(0,0)),mode='constant')
             seq_len=np.pad(seq_len,((0,FLAGS.batch_size-len(seq_len))),mode='constant')
             feed_dict = {x:batch_x,seq_length:seq_len,training:False}
             predict_val = sess.run(predict,feed_dict = feed_dict)
             predict_read = sparse2dense(predict_val)[0]
             if i+FLAGS.batch_size>reads_n:
                 predict_read = predict_read[:reads_n-i]
             reads+=predict_read
         print("Segment reads base calling finished, begin to assembly. %5.2f seconds"%(time.time()-start_time))
         basecall_time=time.time()-start_time
         bpreads = [index2base(read) for read in reads]
         concensus = simple_assembly(bpreads)
         c_bpread = index2base(np.argmax(concensus,axis = 0))
         assembly_time=time.time()-start_time
         print("Assembly finished, begin output. %5.2f seconds"%(time.time()-start_time))
         result_folder = os.path.join(FLAGS.output,'result')
         seg_folder = os.path.join(FLAGS.output,'segments')
         meta_folder = os.path.join(FLAGS.output,'meta')
         if not os.path.exists(FLAGS.output):
             os.makedirs(FLAGS.output)
         if not os.path.exists(seg_folder):
             os.makedirs(seg_folder)
         if not os.path.exists(result_folder):
             os.makedirs(result_folder)
         if not os.path.exists(meta_folder):
             os.makedirs(meta_folder)
         path_con = os.path.join(result_folder,file_pre+'.fasta')
         path_reads = os.path.join(seg_folder,file_pre+'.fasta')
         path_meta=os.path.join(meta_folder,file_pre+'.meta')
         with open(path_reads,'w+') as out_f, open(path_con,'w+') as out_con:
             for indx,read in enumerate(bpreads):
                 out_f.write(">sequence"+str(indx)+'\n')
                 out_f.write(read+'\n')
             out_con.write(">sequence\n"+c_bpread)
         with open(path_meta,'w+') as out_meta:
             total_time = time.time()-start_time
             output_time=total_time-assembly_time
             assembly_time-=basecall_time
             basecall_time-=reading_time                       
             total_len = len(c_bpread)                        
             total_time=time.time()-start_time
             out_meta.write("# Reading Basecalling assembly output total rate(bp/s)\n" )
             out_meta.write("%5.3f %5.3f %5.3f %5.3f %5.3f %5.3f\n"%(reading_time,basecall_time,assembly_time,output_time,total_time,total_len/total_time))
             out_meta.write("#read_len batch_size segment_len jump start_pos\n")
             out_meta.write("%d %d %d %d %d\n"%(total_len,FLAGS.batch_size,FLAGS.segment_len,FLAGS.jump,FLAGS.start))
             out_meta.write("# input_name model_name\n")
             out_meta.write("%s %s\n"%(FLAGS.input,FLAGS.model))
def main():
    time_dict=unix_time(evaluation)
    print FLAGS.output
    print('Real time:%5.3f Systime:%5.3f Usertime:%5.3f'%(time_dict['real'],time_dict['sys'],time_dict['user']))
    meta_folder = os.path.join(FLAGS.output,'meta')
    if os.path.isdir(FLAGS.input):
        file_pre='all'
    else:
        file_pre = os.path.splitext(os.path.basename(FLAGS.input))[0]
    path_meta=os.path.join(meta_folder,file_pre+'.meta')
    with open(path_meta,'a+') as out_meta:
        out_meta.write("$ Wall_time Sys_time User_time Cpu_time\n")
        out_meta.write("%5.3f %5.3f %5.3f %5.3f\n" %(time_dict['real'],time_dict['sys'],time_dict['user'],time_dict['sys']+time_dict['user']))
if __name__=="__main__":
    main()
