#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:57:35 2017

@author: haotianteng
"""

import h5py
import numpy as np
import os,argparse,sys

def extract(FLAGS):
    count = 1
    root_folder = FLAGS.input_dir
    out_folder = FLAGS.output_dir
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if out_folder is None:
        raw_folder = os.path.abspath(os.path.join(out_folder,os.pardir,'raw'))
        ref_folder = os.path.abspath(os.path.join(out_folder,os.pardir,'reference'))
    else:
        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        raw_folder = os.path.abspath(os.path.join(out_folder,'raw'))
        ref_folder = os.path.abspath(os.path.join(out_folder,'reference'))
    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    if not os.path.isdir(ref_folder):
        os.mkdir(ref_folder)
    for file_n in os.listdir(root_folder):
        if file_n.endswith('fast5'):
            try:
                raw_signal,reference = extract_file(root_folder+os.path.sep+file_n)
                count+=1
                if len(raw_signal)==0 or len(reference)==0:
                    continue
            except:
                continue
            signal_file = open(os.path.join(raw_folder,os.path.splitext(file_n)[0]+'.signal'),'w+')
            ref_file = open(os.path.join(ref_folder,os.path.splitext(file_n)[0]+'_ref.fasta'),'w+')
            signal_file.write(" ".join([str(val) for val in raw_signal]))
            ref_file.write(reference)
	    print("Extracted "+(os.path.join(raw_folder,os.path.splitext(file_n)[0]+'.signal')))
def extract_file(input_file):
    try:
        input_data = h5py.File(input_file, 'r')    
    except IOError:
        return False
    except:
        return False
    raw_attr = input_data['Raw/Reads/']
    read_name = raw_attr.keys()[0]
    raw_signal = raw_attr[read_name+'/Signal'].value
    try:
        reference = input_data['Analyses/Basecall_1D_000/BaseCalled_template/Fastq'].value
        reference = '>template\n'+reference.split('\n')[1]
    except:
        reference = input_data['Analyses/Alignment_000/Aligned_template/Fasta'].value
    return raw_signal,reference

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the signal and reference from fast5 file.')
    parser.add_argument('-i','--input_dir', help="Directory that store the fast5 files.")
    parser.add_argument('-o','--output_dir',default = None,help="Directory that output the signal and reference sequence.")
    args = parser.parse_args()
    extract(args)
