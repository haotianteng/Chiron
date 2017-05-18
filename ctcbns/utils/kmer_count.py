#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:37:11 2017

@author: haotianteng
"""
import matplotlib.pyplot as plt
import numpy as np
def kmer2int(kmer):
    base_int={'A':0,'C':1,'G':2,'T':3}
    kmer_arr = [base_int[base] for base in kmer]
    kmer_int = 0
    for base in kmer_arr:
        kmer_int = kmer_int*4+base
    return kmer_int
file_h=open('/home/haotianteng/UQ/deepBNS/ctcbns_package/example_data/Reference/reference.fasta')
kmer_list=list()
for line in file_h:
    if line[0]=='>':
        continue
    line=line[:-1]
    mer_k=5
    for pos,base in enumerate(line):
        kmer=line[pos:pos+mer_k]
        if len(kmer)==mer_k:
            kmer=kmer2int(kmer)
            kmer_list.append(kmer)
            
list_count = [kmer_list.count(i ) for i in range(1024)]