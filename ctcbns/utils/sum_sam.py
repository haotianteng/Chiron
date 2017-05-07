#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  6 00:11:20 2017

@author: haotianteng
"""
import argparse
parser = argparse.ArgumentParser(description='Extract the identity score.')
parser.add_argument('-i','--input', help="Path to .SAM file.")
FLAGS = parser.parse_args()

def count(string):
    count_dict = {'M':0,'I':0,'D':0,'N':0}
    count = ''
    for c in string:
        if c!='M' and c!='I' and c!='D' and c!='N':
            count+=c
        else:
            count_dict[c]+=int(count)
            count=''
    return count_dict['M'],count_dict['I'],count_dict['D'],count_dict['N']
def main():
    sam_f = open(FLAGS.input,'r')
    for line in sam_f:
        line = line.split()
        if line[0]=='sequence':
            Alignment = line[5]
            M,I,D,N=count(Alignment)
            print M,I,D,float(M)/(M+I+D+N)
if __name__=='__main__':
    main()