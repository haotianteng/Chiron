#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:38:47 2018

@author: heavens
"""
import argparse
import sys
import collections
from tqdm import tqdm
MAX_CHUNK_SIZE = 5e8
def fast_reader(in_file,out_file):
    seqs = collections.OrderedDict()
    with open(in_file,'r') as f:
        for line in tqdm(f,desc = 'Read line in fast file:'):
            if line.startswith('>') or line.startswith('@'):
                last_seq = line.strip()
                seqs[last_seq] = ''
            else:
                seqs[last_seq] +=line
    with open(out_file,'w') as out_f:
        for k,v in seqs.items():
            out_f.write(k+'\n'+v)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer fast5 to raw_pair file.')
    parser.add_argument('-i', '--input', required = True,
                        help="Input fasta(q) file.")
    parser.add_argument('-o', '--output', required = True, help="Output fasta(q) file")
    args = parser.parse_args(sys.argv[1:])
    fast_reader(args.input,args.output)
