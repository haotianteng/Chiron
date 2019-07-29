#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:29:40 2018

@author: heavens
"""
import argparse
import sys
import h5py
import os
import numpy as np
import difflib
import bisect
from scipy.interpolate import interp1d
import itertools
import mappy
from tqdm import tqdm
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from multiprocessing import Pool
OVERMOVE_ERROR = "Encounter a movement bigger than 4!"
NEGTIVE_ERROR = "Negative movement detected."

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
    Input Args:
        bound: A array with shape [N,2], the first occurance and last occurance position of the base.
    Return:
        A array with shape [N,2], the interpolated location of the LIS sequence of the positions of each base.
    """
    bot_val, bot_idx = LIS(bound[:,0])
    up_val, up_idx = LIS(bound[:,1])
    bot_f = interp1d(bot_idx,bot_val)
    up_f = interp1d(up_idx,up_val)
    up_bound = up_f(range(len(bound)))
    bot_bound = bot_f(range(len(bound)))
    return np.asarray(list(zip(bot_bound,up_bound)))

def match_blocks(alignment):
    tmp_start = -1 
    blocks = []
    pos_0 = 0
    pos_1 = 0
    for idx,base in enumerate(alignment[0]):
        if (alignment[0][idx] == '-') or (alignment[1][idx] == '-'):
            if tmp_start >= 0:
                blocks.append([idx - tmp_start,pos_0,pos_1])
                tmp_start = -1
        else:
            if tmp_start == -1:
                tmp_start = idx
        if alignment[0][idx] != '-':
            pos_0 += 1
        if alignment[1][idx] != '-':
            pos_1 += 1
    if tmp_start >=0:
        blocks.append([idx - tmp_start,pos_0,pos_1])
    return blocks

def global_alignment_assembly_pos(bpreads):
    concensus = np.zeros([4, 1000])
    concensus_bound = np.zeros([4,1000,2])
    concensus_bound[:,:,0] = np.inf
    pos_collection = [[0,len(bpreads[0])]]
    pos = 0
    length = 0
    census_len = 1000
    gap_open = -5
    gap_extend = -2
    mismatch = -3
    match = 1
    min_block_size = 3
    for idx, bpread in enumerate(bpreads):
        disp = None
        if idx == 0:
            add_bound(concensus,concensus_bound, 0, bpread,idx)
            continue
        prev_bpread = bpreads[idx - 1]
        global_alignment = pairwise2.align.globalms(prev_bpread,bpread,match,mismatch,gap_open,gap_extend)
        if len(global_alignment) == 0:
            continue
        blocks = match_blocks(global_alignment[0])
        for block in blocks:
            if block[0] >= min_block_size:
                disp = block[1] - block[2]
                break
        if disp is None:
            disp = blocks[0][1] - blocks[0][2]
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
        length = max(length, pos + len(bpread))    
    return concensus[:, :length],concensus_bound[:, :length,:],pos_collection
  
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
    concensus,bound,coors = global_alignment_assembly_pos(chunks)
#    concensus,bound,coors = simple_assembly_pos(chunks,0.1)
    c_indexs = np.argmax(concensus,axis = 0)
    bound = bound[c_indexs,np.arange(bound.shape[1]),:]
    bound  = get_squiggle_pos(bound)
    loc = np.mean(bound,axis = 1)
    return chunks,bound,loc,concensus,coors

def revise(concensus, ref_file):
    pass
### Test Script ###
    
def reformat_hmm(fast5_f):
    DATA_FORMAT = np.dtype([('mean','<f4'),
                            ('std','<f4'),
                            ('start','<i4'),
                            ('length','<i4'),
                            ('base','S1')])
    event_entry_id = "/Analyses/AlignToRef_000/CurrentSpaceMapped_template/Events"
    with h5py.File(fast5_f,'r+') as root:
        events = root[event_entry_id].value
        raw_entry = list(root['/Raw/Reads'].values())[0]
        start_time = raw_entry.attrs['start_time']
        raw_signal = raw_entry['Signal'].value
        sample_rate = int(root["/UniqueGlobalKey/context_tags"].attrs['sample_frequency'])
        start_int = np.round(events['start'] *sample_rate).astype(int) - start_time
        total_len = len(raw_signal)
        start = list()
        rev_start = list()
        length = list()
        base = list()
        means = list()
        stds = list()
        for idx,pos in enumerate(events['seq_pos']):
            if idx == 0:
                prev_start = start_int[idx]
                prev_kmer = events['kmer'][idx].decode("utf-8")
                prev_pos = pos
            elif pos > prev_pos:
                curr_start = start_int[idx]
                move = pos - prev_pos
                if move > 4:
                    raise ValueError(OVERMOVE_ERROR)
                prev_kmer = prev_kmer + events['kmer'][idx].decode("utf-8")[-move:]
                avg_len = int(round((curr_start - prev_start)/move))
                for i in range(move):
                    start.append(prev_start)
                    length.append(avg_len)
                    prev_start = prev_start + avg_len
                    base.append(prev_kmer[2+i])
                    chunk = raw_signal[start[-1]:(start[-1] + length[-1])]
                    means.append(np.mean(chunk))
                    stds.append(np.std(chunk))
                length[-1] = curr_start - start[-1]
                prev_pos = pos
                prev_start = curr_start
                prev_kmer = events['kmer'][idx].decode("utf-8")
            elif pos < prev_pos:
                raise ValueError(NEGTIVE_ERROR)
        for idx,s in enumerate(start):
            rev_start.append(total_len - s - length[idx])
        matrix = list()
        rev_start = rev_start[::-1]
        length = length[::-1]
        base = base[::-1]
        means = means[::-1]
        stds = stds[::-1]
        matrix = np.asarray(list(zip(means,stds,rev_start,length,base)),dtype = DATA_FORMAT)
        if '/Analyses/RawGenomeCorrected_000' in root:
            del root['/Analyses/RawGenomeCorrected_000']
        event_h = root.create_dataset('/Analyses/RawGenomeCorrected_000/BaseCalled_template/Events', shape = (len(matrix),),maxshape=(None,),dtype = DATA_FORMAT)
        event_h[...] = matrix
        event_h.attrs['read_start_rel_to_raw'] = 0

def wrapper_reformat_hmm(args):
    fast5_f, fail_count = args
    try:
        reformat_hmm(fast5_f)
        return('Succeed')
    except ValueError as e:
        return(str(e))
def run(args):
    fail_count = {OVERMOVE_ERROR:0, NEGTIVE_ERROR:0,'Succeed':0}
    file_list = []
    for file in os.listdir(args.source):
        if file.endswith('fast5'):
            file_list.append(os.path.join(args.source,file)) 
    pool = Pool(args.thread)
    for state in tqdm(pool.imap_unordered(wrapper_reformat_hmm,zip(file_list,itertools.repeat(fail_count))),total = len(file_list)):
        if state in fail_count.keys():
            fail_count[state] +=1
        else:
            fail_count[state] = 1
    pool.close()
    pool.join() 
    print(fail_count)       
###################

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(
#        description='Transfer fast5 to raw_pair file.')
#    parser.add_argument('-s', '--source', required = True,
#                        help="Directory that store the output subfolders.")
#    parser.add_argument('-t', '--thread', default = 1, type = int,
#                        help="Thread number used.")
##    parser.add_argument('-d', '--dest', required = True, 
##                        help="Folder that contain fast5 files to resquiggle")
##    parser.add_argument('--basecall_group',default = "Chiron_Basecall_1D_000",
##                        help='The attribute group to resquiggle the training data in.')
##    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
##                        help='Basecall subgroup ')
##    parser.add_argument('--mode',default = 'dna',
##                        help='Type of data to resquiggle, default is dna, can be chosen from dna or rna.')
#    args = parser.parse_args(sys.argv[1:])
#    run(args)   chunks
    
    ### Test Code ###
LIS([1,8,3,4,5,2])
ROOT_FOLDER = "/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_GN131/test/"
FAST5_FOLDER = "/home/heavens/UQ/Chiron_project/RNA_Analysis/RNA_GN131/test/"
FILE_PRE = "imb17_013486_20171113_FAB45360_MN17279_sequencing_run_20171113_RNAseq_GN131_17776_read_1002_ch_242_strand"
FILE_PRE = "imb17_013486_20171113_FAB45360_MN17279_sequencing_run_20171113_RNAseq_GN131_17776_read_11842_ch_59_strand"
REF_FILE = "/home/heavens/UQ/Chiron_project/RNA_Analysis/Reference/S00000028.fasta"

chunks,bounds,locs,concensus,coors = resquiggle(ROOT_FOLDER, FAST5_FOLDER, FILE_PRE)
from matplotlib import pyplot as plt
chunk_size = len(chunks)
for idx,_ in enumerate(bounds):
    plt.axvline(x = idx, ymin = bounds[idx,0]/chunk_size, ymax = bounds[idx,1]/chunk_size)
plt.plot(np.arange(len(locs)),locs)
plt.yticks(np.arange(0,chunk_size,chunk_size/10))
gap_open = -5
gap_extend = -2
mismatch = -3
match = 1
global_alignment = pairwise2.align.globalms(chunks[0],chunks[1],match,mismatch,gap_open,gap_extend)
print(format_alignment(*global_alignment[0]))
match_blocks(global_alignment[0])
    
print("#################################")
#coors = np.asarray(coors)
#con_len = len(concensus[0])
#for idx,_ in enumerate(coors):
#    plt.axhline(y = idx, xmin = coors[idx,0]/float(con_len), xmax = coors[idx,1]/float(con_len))
    ###

