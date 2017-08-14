#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 18:38:18 2017

@author: haotianteng
"""
import argparse
import chiron_eval
import chiron_rcnn_train
from utils.extract_sig_ref import extract
def evaluation(args):
    chiron_eval(args)
def main():
    parser=argparse.ArgumentParser(prog='chiron',description='A deep neural network basecaller.')
    subparsers = parser.add_subparsers(title='111',help='111')
    #parser for 'extract' command
    #parser for 'call' command
    parser_call=subparsers.add_parser('call',title='call_title', description='Perform basecalling',help='call help')
    parser_call.add_argument('-i','--input', help="File path or Folder path to the fast5 file.")
    parser_call.add_argument('-o','--output', help = "Output folder path")
    parser_call.add_argument('-m','--model', default = '../model/crnn3+3_S10_2_re',help = "model folder")
    parser_call.add_argument('-s','--start',type=int,default = 0,help = "Start index of the signal file.")
    parser_call.add_argument('-b','--batch_size',type = int,default = 1100,help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
    parser_call.add_argument('-l','--segment_len',type = int,default = 300, help="Segment length to be divided into.")
    parser_call.add_argument('-j','--jump',type = int,default = 30,help = "Step size for segment")
    parser_call.add_argument('-t','--threads',type = int,default = 0,help = "Threads number")
    parser_call.set_defaults(func=chiron_eval.run)
    
    #parser for 'train' command
    parser_train=subparsers.add_parser('train',title='train_title',description='Model training',help='train help')
    parser_train.add_argument('-i','--data_dir',help="Fast5 file with labelled data.")
    parser_train.add_argument('-o','--log_dir',help="Log dir which save the trained model")
    parser_train.add_argument('-n','--model_name',help="Model name saved.")
    parser_train.add_argument('-t','--retrain',type='bool',default=False,help="If retrain is true, the previous trained model will be loaded before training.")
    parser_train.add_argument('-l','--sequence_len',type=int,default=300,help="Segment length to be divided into.")
    parser_train.add_argument('-b','--batch_size',type=int,default=300,help="Batch size to train, large batch size require more ram but give faster training speed.")
    parser_train.add_argument('-m','--max_steps',type=int,default=20000,help="Maximum training steps conducted.")
    parser_train.add_argument('-r','--step_rate',type=float,default=1e-3,help="Learning rate used for optimiztion algorithm.")
    parser_train.add_argument('-k','--k_mer',type=int,default=1,help="Output k-mer size.")
    parser_train.set_defaults(func=chiron_rcnn_train.run)
    
    args=parser.parse_args()
    args.func(args)
    