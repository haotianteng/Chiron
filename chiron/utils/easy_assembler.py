#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 15:39:29 2017

@author: haotianteng
"""
import numpy as np
import difflib

def simple_assembly(bpreads):
    concensus = np.zeros([4,1000])
    pos = 0
    length = 0
    census_len = 1000
    for indx,bpread in enumerate(bpreads):
        if indx==0:
            add_count(concensus,0,bpread)
            continue
        d = difflib.SequenceMatcher(None,bpreads[indx-1],bpread)
        match_block = max(d.get_matching_blocks(),key = lambda x:x[2])
        disp = match_block[0]-match_block[1]
        if disp+pos+len(bpreads[indx])>census_len:
            concensus = np.lib.pad(concensus,((0,0),(0,1000)),mode = 'constant',constant_values = 0)
            census_len +=1000
        add_count(concensus,pos+disp,bpreads[indx])
        pos+=disp
        length = max(length,pos+len(bpreads[indx]))
    return concensus[:,:length]
    
def add_count(concensus,start_indx,segment):
    base_dict = {'A':0,'C':1,'G':2,'T':3,'a':0,'c':1,'g':2,'t':3}
    if start_indx<0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i,base in enumerate(segment):
        concensus[base_dict[base]][start_indx+i] += 1

def main():
    bpreads = ['AAGGCCTAGCT','AGGCCTAGCAA','GGCCTAGCTC','AAAGGCCTAGT']
    census,length = simple_assembly(bpreads)
    result = np.argmax(census,axis=0)
    print result
if __name__=='__main__':
    main()