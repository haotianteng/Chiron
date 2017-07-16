#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 20:15:12 2017

@author: haotianteng
"""

import os,argparse,sys
parser = argparse.ArgumentParser(description='Transfer signal_label to raw_pair file.')
parser.add_argument('-i','--input_dir', help="Directory that store the signal_label file.")
args = parser.parse_args()

def extract(raw_folder = None):
    count = 1
    root_folder = args.input_dir
    print(root_folder)
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if raw_folder is None:
        raw_folder = os.path.abspath(os.path.join(root_folder, os.pardir))+'/raw_ws/'
    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    for file_n in os.listdir(root_folder):
        success = False
        print(file_n)
        if file_n.endswith('signal_label'):
            output_file = raw_folder+os.path.splitext(file_n)[0]
            print(output_file)
            try:
                success = extract_file(root_folder+os.path.sep+file_n,output_file)
            except:
                continue
            if success:
                count +=1
            sys.stdout.write("%s file transfered.   \n" % (file_n) )
def extract_file(input_file,output_file):
    with open(input_file,'r') as f:
        raw_signal = list()
        raw_labels = list()        
        for line in f:
            line_sp  = line.split()
            raw_signal.append(line_sp[0])
            raw_labels.append(line_sp[1])
    raw_start,raw_end,raw_kmer = read_label(raw_labels)            
    f_signal = open(output_file+'.signal','w+')
    f_label = open(output_file+'.label','w+')
    f_label_plain = open(output_file+'.plain_label','w+')
    for value in raw_signal:
        f_signal.write("%s\n"%(value))
    f_label_plain.write("".join([kmer2base(x) for x in raw_kmer]))
    for index,start in enumerate(raw_start):    
        f_label.write("%d %d %d\n"%(start,raw_end[index],raw_kmer[index]))
    f_signal.close()
    f_label.close()
    return True
def kmer2base(kmer):
    base = kmer/16%4
    base_list = ['A','C','G','T']
    return base_list[base]
def read_label(labels):
    start = list()
    end = list()
    k_mer = list()
    if type(labels[0])!=type(1):
        labels = [int(x) for x in labels]
    labels_len = len(labels)
    for pos,label in enumerate(labels):
        if pos < labels_len-2:
            next_label = labels[pos+1]
        else:
            if label!=-1:
                end.append(pos)
                k_mer.append(label)
        if label!=next_label:
             if label!=-1:
                 end.append(pos)
                 k_mer.append(label)
             if next_label!=-1:
                 start.append(pos+1)
    return start,end,k_mer

            
def main():
    extract()
            
if __name__ == '__main__':
    root_folder = '/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4'
    main()
