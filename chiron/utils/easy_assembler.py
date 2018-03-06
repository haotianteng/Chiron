# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Tue May  2 15:39:29 2017

from __future__ import absolute_import
from __future__ import print_function
import difflib
import math
import operator
import time
from collections import Counter
from itertools import groupby

import numpy as np
import six
from six.moves import range


def mapping(full_path, blank_pos=4):
    """Perform a many to one mapping in the CTC paper, merge the repeat and remove the blank
    Input:
        full_path:a vector of path, e.g. [1,0,3,2,2,3]
        blank_pos:The number regarded as blank"""
    full_path = np.asarray(full_path)
    merge_repeated = np.asarray([k for k, g in groupby(full_path)])
    blank_index = np.argwhere(merge_repeated == blank_pos)
    return np.delete(merge_repeated, blank_index)


def list2string(input_v, base_type):
    if base_type == 0:
        base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'b'}
    if base_type == 1:
        base_dict = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return "".join(base_dict[item] for item in input_v)


def string2list(input_v, base_type):
    if base_type == 0:
        base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    result = list()
    for item in input_v:
        result.append(base_dict[item])
    return result


def group_consecutives(vector, step=1):
    group = list()
    group_list = list()
    expect = None
    for x in vector:
        if (x != expect) and (expect is not None):
            group_list.append(group)
            group = []
        group.append(x)
        expect = x + step
    group_list.append(group)
    return group_list


###########################Section decoding method#############################
def section_decoding(logits, blank_thres=0.6, base_type=0):
    """Implemented the decoding method described in ftp://ftp.idsia.ch/pub/juergen/icml2006.pdf
    Find the best path between the section that divided by blank logits < 0.9
    
    logits: [batch_size,seg_length,neucloe_type+1]
    base_type: 0:dna 1:methylation 2:rna
    """
    prob = np.exp(logits) / (np.sum(np.exp(logits), axis=2)[:, :, None])
    if base_type == 0:
        blank_pos = 4
    mask = prob[:, :, blank_pos] < blank_thres
    batch_size, seg_len, nc_type = prob.shape
    bpreads = list()
    bpread = list()
    for batch_i in range(batch_size):

        group_list = group_consecutives(np.where(mask[batch_i, :])[0])

        bpread = []
        for group in group_list:
            if len(group) == 0:
                continue
            bpread.append(4)
            #            most_prob_path = best_path(prob[batch_i,group,:],base_type = base_type)
            most_mc_path = mc_path(logits[batch_i, group, :],
                                   base_type=base_type)
            most_prob_path = string2list(most_mc_path[0])
            bpread += most_prob_path
        bpreads.append(list2string(mapping(bpread), base_type=base_type))
    return (bpreads)


def best_path(logits, base_type):
    """Enumerate decoder,*slow*"""
    T, base_num = logits.shape
    accum_prob = {}
    for i in range(base_num ** T):
        prob = 1
        index_list = []
        for j in range(T):
            index = i / base_num ** j % base_num
            prob *= logits[j, index]
            index_list.append(index)
        index_list = mapping(index_list)
        if len(index_list) > 0:
            key = list2string(index_list, base_type=base_type)
            accum_prob.setdefault(key, 0)
            accum_prob[key] += prob
    most_prob_path = max(six.iteritems(accum_prob), key=operator.itemgetter(1))[
        0]
    return string2list(most_prob_path, base_type=base_type)


def mc_path(logits, base_type, sample_n=300):
    """Manto Carlo decoder
    Input Args:
        logits:[T,base_num]
        base_tyep: 0:normal dna+blank
        sample_n: Times of sample used in the Manto Carlo simulation.
    """
    logits_shape = logits.shape
    bpreads = list()
    qc_score = list()
    prob = np.exp(logits) / (np.sum(np.exp(logits), axis=2)[:, :, None])
    base_num = logits_shape[-1]
    T = logits_shape[-2]
    interval = np.zeros((T, base_num))
    interval[:, 0] = prob[:, 0]
    for i in range(1, base_num - 1):
        interval[:, i] = interval[:, i - 1] + prob[:, i]
    interval[:, 4] = 1

    sample_index = np.zeros((sample_n, T))
    sample = np.random.random((sample_n, T))
    for j in range(T):
        sample_index[:, j] = np.searchsorted(interval[i, j, :], sample[:, j],
                                             side='left')
    merge_path = list()
    for repeat_i in range(sample_n):
        ###This step may be slow, considering implemented in C
        temp_path = mapping(sample_index[repeat_i, :])
        ###
        merge_path.append(list2string(temp_path, base_type=base_type))
    path_count = Counter(merge_path)
    print(path_count)
    max2path = path_count.most_common(2)
    p1 = max2path[0][1] / float(sample_n)
    p2 = max2path[1][1] / float(sample_n)
    qc_score.append(10 * math.log10(p1 / p2))
    bpreads.append(max2path[0][0])
    return bpreads


def mc_decoding(logits, base_type, sample_n=300):
    """Manto Carlo decoder
    Input Args:
        logits:[batch_size,T,base_num] or [T,base_num]
        base_tyep: 0:normal dna+blank
        sample_n: Times of sample used in the Manto Carlo simulation.
    """
    logits_shape = logits.shape
    if len(logits_shape) == 2:
        logits = [logits]
        batch_size = 1
    else:
        batch_size = logits_shape[0]
    bpreads = list()
    qc_score = list()
    prob = np.exp(logits) / (np.sum(np.exp(logits), axis=2)[:, :, None])
    base_num = logits_shape[-1]
    T = logits_shape[-2]
    interval = np.zeros((batch_size, T, base_num))
    interval[:, :, 0] = prob[:, :, 0]
    for i in range(1, base_num - 1):
        interval[:, :, i] = interval[:, :, i - 1] + prob[:, :, i]
    interval[:, :, 4] = 1

    sample_index = np.zeros((sample_n, T))
    for i in range(batch_size):
        print(i)
        sample = np.random.random((sample_n, T))
        for j in range(T):
            sample_index[:, j] = np.searchsorted(interval[i, j, :],
                                                 sample[:, j], side='left')
        merge_path = list()
        for repeat_i in range(sample_n):
            ###This step may be slow, considering implemented in C
            temp_path = mapping(sample_index[repeat_i, :])
            ###
            merge_path.append(list2string(temp_path, base_type=base_type))
        path_count = Counter(merge_path)
        print(path_count)
        max2path = path_count.most_common(2)
        p1 = max2path[0][1] / float(sample_n)
        p2 = max2path[1][1] / float(sample_n)
        qc_score.append(10 * math.log10(p1 / p2))
        bpreads.append(max2path[0][0])
    return bpreads


###############################################################################

#########################Simple assembly method################################
def simple_assembly(bpreads):
    concensus = np.zeros([4, 1000])
    pos = 0
    length = 0
    census_len = 1000
    for indx, bpread in enumerate(bpreads):
        if indx == 0:
            add_count(concensus, 0, bpread)
            continue
        d = difflib.SequenceMatcher(None, bpreads[indx - 1], bpread)
        match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
        disp = match_block[0] - match_block[1]
        if disp + pos + len(bpreads[indx]) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            census_len += 1000
        add_count(concensus, pos + disp, bpreads[indx])
        pos += disp
        length = max(length, pos + len(bpreads[indx]))
    return concensus[:, :length]


def add_count(concensus, start_indx, segment):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base_dict[base]][start_indx + i] += 1


###############################################################################

#########################Simple assembly method with quality score################################
def simple_assembly_qs(bpreads, qs_list):
    concensus = np.zeros([4, 1000])
    concensus_qs = np.zeros([4, 1000])
    pos = 0
    length = 0
    census_len = 1000
    assert len(bpreads) == len(qs_list)
    for indx, bpread in enumerate(bpreads):
        if indx == 0:
            add_count_qs(concensus, concensus_qs, 0, bpread, qs_list[indx])
            continue
        d = difflib.SequenceMatcher(None, bpreads[indx - 1], bpread)
        match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
        disp = match_block[0] - match_block[1]
        if disp + pos + len(bpread) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            concensus_qs = np.lib.pad(concensus_qs, ((0, 0), (0, 1000)),
                                      mode='constant', constant_values=0)
            census_len += 1000
        add_count_qs(concensus, concensus_qs, pos + disp, bpread, qs_list[indx])
        pos += disp
        length = max(length, pos + len(bpread))
    return concensus[:, :length], concensus_qs[:, :length]


def add_count_qs(concensus, concensus_qs, start_indx, segment, qs):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base_dict[base]][start_indx + i] += 1
        concensus_qs[base_dict[base]][start_indx + i] += qs[0]


###############################################################################

def main():
    # bpreads = ['AAGGCCTAGCT','AGGCCTAGCAA','GGCCTAGCTC','AAAGGCCTAGT']
    #    logits_sample = np.load('/home/haotianteng/UQ/deepBNS/Chiron_Project/chiron_fastqoutput/chiron/utils/logits_sample.npy')
    start = time.time()
    # test = mc_path(logits_sample[300,:,:],base_type = 0)
    # print time.time()-start
#    bpreads = section_decoding(logits_sample)
# census = simple_assembly(bpreads)

#    result = np.argmax(census,axis=0)
#    print result
