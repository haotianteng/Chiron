#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:29:40 2018

@author: heavens
"""
#import argparse
#import sys
#import h5py
import os
import numpy as np
import difflib
import mappy
import bisect

def LIS(sequence):
    """
    This function finding the longest increasing subsequence for a given sequence.
    An implemention of the binary search method in Wiki:
        https://en.wikipedia.org/wiki/Longest_increasing_subsequence
    """
    sequence = np.asarray(sequence)
    l = 1
    M = np.array([0])
    P = np.array([-1]*len(sequence))
    for i,item in enumerate(sequence[1:]):
        idx = i+1
        if item > sequence[M[-1]]:
            P[idx] = M[-1]
            M = np.append(M,idx)
            l += 1
        else:
            loc = bisect.bisect_left(sequence[M],sequence[idx])
            P[idx] = M[loc-1]
            M[loc] = idx
    out = [-1] * l
    index = M[l-1]
    idxs = [index]
    for i in range(l-1,-1,-1):
        out[i] = sequence[index]
        index = P[index]
        idxs = [index] + idxs
    return out,idxs[1:]
def simple_assembly_pos(bpreads):
    concensus = np.zeros([4, 1000])
    concensus_bound = np.zeros([4,1000,2])
    concensus_bound[:,:,0] = np.inf
    pos = 0
    length = 0
    census_len = 1000
    for idx, bpread in enumerate(bpreads):
        if idx == 0:
            add_bound(concensus,concensus_bound, 0, bpread,idx)
            continue
        d = difflib.SequenceMatcher(None, bpreads[idx - 1], bpread)
        match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
        disp = match_block[0] - match_block[1]
        if disp + pos + len(bpread) > census_len:
            concensus = np.lib.pad(concensus, 
                                   ((0, 0), (0, 1000)),
                                   mode='constant', 
                                   constant_values=0)
            concensus_bound = np.lib.pad(concensus_bound,
                                         ((0, 0), (0, 1000),(0,0)),
                                         mode='constant', 
                                         constant_values=0)
            concensus_bound[:,census_len:census_len+1000,0] = np.inf
            census_len += 1000
        add_bound(concensus,concensus_bound, pos + disp, bpread,idx)
        pos += disp
        length = max(length, pos + len(bpread))
    return concensus[:, :length],concensus_bound[:, :length,:]


def add_bound(concensus,concensus_bound, start_indx, segment,segment_idx):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'U':3, 
                 'a': 0, 'c': 1, 'g': 2, 't': 3, 'u':3}
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        if base == 'X':
            print(concensus)
            print(segment)
        concensus[base_dict[base]][start_indx + i] += 1
        concensus_bound[base_dict[base]][start_indx + i][0] = min(concensus_bound[base_dict[base]][start_indx + i][0],segment_idx)
        concensus_bound[base_dict[base]][start_indx + i][1] = max(concensus_bound[base_dict[base]][start_indx + i][1],segment_idx)

def read_chunks(filepath):
    chunks = list()
    with open(filepath,'r') as f:
        for line in f:
            if line.startswith(">") or line.startswith("@"):
                pass
            else:
                chunks.append(line.strip())
    return chunks
def read_meta(filepath):
    keys = []
    meta_dict = {}
    with open(filepath,'r') as f:
        for line in f:
            if line.startswith("#"):
                keys+= line[1:].split()
            else:
                if keys[0] == "read_len":
                    for idx,val in enumerate(line[1:].split()):
                        meta_dict[keys[idx]] = int(val.strip())
                keys = []
    return meta_dict

def write_back_to_fast5():
    pass 

def resquiggle(root_folder,fast5_folder,file_pre):
    meta_path = os.path.join(root_folder,'meta',file_pre+'.meta')
    chunk_path = os.path.join(root_folder,'segments',file_pre+'.fastq')
    fast5_path = os.path.join(fast5_folder,file_pre+'.fast5')
    if not os.path.isfile(meta_path):
        raise ValueError("Metaparameter file not found")
    if not os.path.isfile(chunk_path):
        raise ValueError("Segments file not found")
    chunks = read_chunks(chunk_path)
    meta_pre= read_meta(meta_path)
    concensus,bound = simple_assembly_pos(chunks)
    c_indexs = np.argmax(concensus,axis = 0)
    bound = bound[c_indexs,np.arange(bound.shape[1]),:]
    loc = np.mean(bound,axis = 1)
    return chunks,bound,loc

### Test Script ###
LIS([1,8,3,4,5,2])
ROOT_FOLDER = "/home/heavens/UQ/Chiron_project/test_data/output/"
FAST5_FOLDER = "/home/heavens/UQ/Chiron_project/test_data/output/fast5s"
FILE_PRE = "IMB14_011406_LT_20170322_FNFAF13375_MN17027_sequencing_run_C4_watermang_22032017_12981_ch2_read2558_strand"

chunks,bounds,locs = resquiggle(ROOT_FOLDER, FAST5_FOLDER, FILE_PRE)
from matplotlib import pyplot as plt
chunk_size = len(chunks)
#for idx,_ in enumerate(bounds):
#    plt.axvline(x = idx, ymin = bounds[idx,0]/chunk_size, ymax = bounds[idx,1]/chunk_size)
#plt.yticks(np.arange(0,400,30))
plt.plot(np.arange(len(locs)),locs)
###################

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(
#        description='Transfer fast5 to raw_pair file.')
#    parser.add_argument('-s', '--source', required = True,
#                        help="Directory that store the output subfolders.")
#    parser.add_argument('-d', '--dest', required = True, 
#                        help="Folder that contain fast5 files to resquiggle")
#    parser.add_argument('--basecall_group',default = "Chiron_Basecall_1D_000",
#                        help='The attribute group to resquiggle the training data in.')
#    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
#                        help='Basecall subgroup ')
#    parser.add_argument('--mode',default = 'dna',
#                        help='Type of data to resquiggle, default is dna, can be chosen from dna or rna.')
#    args = parser.parse_args(sys.argv[1:])
