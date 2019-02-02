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
from scipy.interpolate import interp1d
import itertools

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
    idxs = idxs[1:]
    if idxs[0]!=0:
        out = [sequence[0]]+out
        idxs = [0] + idxs
    if idxs[-1]!=len(sequence)-1:
        idxs = idxs + [len(sequence)-1]
        out = out + [sequence[-1]]
    return np.asarray(out),np.asarray(idxs)

def get_squiggle_pos(bound):
    """
    Get the position of the persudo squiggles by interpolating the bounds.
    """
    bot_val, bot_idx = LIS(bound[:,0])
    up_val, up_idx = LIS(bound[:,1])
    bot_f = interp1d(bot_idx,bot_val)
    up_f = interp1d(up_idx,up_val)
    up_bound = up_f(range(len(bound)))
    bot_bound = bot_f(range(len(bound)))
    return np.asarray(list(zip(bot_bound,up_bound)))
    
def simple_assembly_pos(bpreads,jump_step_ratio, error_rate = 0.2):
    """
    Assemble the read from the chunks. Log probability is 
    log_P ~ x*log((N*n1/L)) - log(x!) + Ns * log(P1/0.25) + Nd * log(P2/0.25)
    Args:
        bpreads: Input chunks.
        jump_step_ratio: Jump step divided by segment length.
        error_rate: An estimating basecalling error rate.
    """
    concensus = np.zeros([4, 1000])
    concensus_bound = np.zeros([4,1000,2])
    concensus_bound[:,:,0] = np.inf
    pos_collection = [[0,len(bpreads[0])]]
    pos_log_p = [1]
    pos = 0
    length = 0
    census_len = 1000
    back_ratio = 6.5 * 10e-4
    p_same = 1 - 2*error_rate + 26/25*(error_rate**2)
    p_diff = 1 - p_same
    for idx, bpread in enumerate(bpreads):
        if idx == 0:
            add_bound(concensus,concensus_bound, 0, bpread,idx)
            continue
        prev_bpread = bpreads[idx - 1]
        ns = dict() # number of same base
        nd = dict()
        log_px = dict()
        N = len(bpread)
        match_blocks = difflib.SequenceMatcher(a=bpread,b=prev_bpread).get_matching_blocks()
        for idx, block in enumerate(match_blocks):
            offset = block[1] - block[0]
            if offset in ns.keys():
                ns[offset] = ns[offset] + match_blocks[idx][2]
            else:
                ns[offset] = match_blocks[idx][2]
            nd[offset] = 0
        
#        for offset in range(-3,len(prev_bpread)):
#            pair = itertools.zip_longest(prev_bpread[offset:],bpread,fillvalue=None)
#            comparison = [int(i==j) for i,j in pair]
#            ns[offset] = sum(comparison)
#            nd[offset] = len(comparison) - ns[offset]
        for key in ns.keys():
            if key < 0:
                k = -key
                log_px[key] = k*np.log((back_ratio)*N*jump_step_ratio) - sum([np.log(x+1) for x in range(k)]) +\
                ns[key]*np.log(p_same/0.25) + nd[key]*np.log(p_diff/0.25)
            else:
                log_px[key] = key*np.log(N*jump_step_ratio) - sum([np.log(x+1) for x in range(key)]) +\
                ns[key]*np.log(p_same/0.25) + nd[key]*np.log(p_diff/0.25)
        disp = max(log_px.keys(),key = lambda x: log_px[x])
#        disp = match_block[0] - match_block[1]
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
        pos_collection.append([pos,pos+len(bpread)])
        pos_log_p.append(log_px[disp])
        length = max(length, pos + len(bpread))
    return concensus[:, :length],concensus_bound[:, :length,:],pos_collection


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

def get_events(consensus, aligner):
    pass

def write_fast5(locs, fast5_h, consensus, ):
    """
    Write the event matrix back to fast5 files.
    """
    pass 

def resquiggle(root_folder,fast5_folder,file_pre):
    meta_path = os.path.join(root_folder,'meta',file_pre+'.meta')
    chunk_path = os.path.join(root_folder,'segments',file_pre+'.fastq')
#    fast5_path = os.path.join(fast5_folder,file_pre+'.fast5')
    if not os.path.isfile(meta_path):
        raise ValueError("Metaparameter file not found")
    if not os.path.isfile(chunk_path):
        raise ValueError("Segments file not found")
    chunks = read_chunks(chunk_path)
    metainfo= read_meta(meta_path)
    concensus,bound,coors = simple_assembly_pos(chunks,0.1)
    c_indexs = np.argmax(concensus,axis = 0)
    bound = bound[c_indexs,np.arange(bound.shape[1]),:]
    bound  = get_squiggle_pos(bound)
    loc = np.mean(bound,axis = 1)
    return chunks,bound,loc,concensus,coors

def revise(concensus, ref_file):
    pass
### Test Script ###
    
LIS([1,8,3,4,5,2])
ROOT_FOLDER = "/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_GN131/test/"
FAST5_FOLDER = "/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_GN131/test/"
#FILE_PRE = "imb17_013486_20171113_FAB45360_MN17279_sequencing_run_20171113_RNAseq_GN131_17776_read_1002_ch_242_strand"
FILE_PRE = "imb17_013486_20171113_FAB45360_MN17279_sequencing_run_20171113_RNAseq_GN131_17776_read_1002_ch_242_strand"

chunks,bounds,locs,concensus,coors = resquiggle(ROOT_FOLDER, FAST5_FOLDER, FILE_PRE)
#from matplotlib import pyplot as plt
#chunk_size = len(chunks)
#for idx,_ in enumerate(bounds):
#    plt.axvline(x = idx, ymin = bounds[idx,0]/chunk_size, ymax = bounds[idx,1]/chunk_size)
#plt.plot(np.arange(len(locs)),locs)
#plt.yticks(np.arange(0,chunk_size,chunk_size/10))

coors = np.asarray(coors)
con_len = len(concensus[0])
for idx,_ in enumerate(coors):
    plt.axhline(y = idx, xmin = coors[idx,0]/float(con_len), xmax = coors[idx,1]/float(con_len))
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
