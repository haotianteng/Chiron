# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys 
from subprocess import call
import mappy as mp
import argparse
import re
class alignments:
    def __init__(self):
        """
        names: sequences names
        cigar: cigar strings, the reverse completement string when the strand is reverse aligned.
        r_st: start position of the reference genome
        r_end: end position of the reference genome
        q_st: start position of the query genome
        q_end: end position of the query genome
        section: the name of the section, in Eukaryotes is the chromesome name
        strand: 1 for forward strand, -1 for reverese strand
        """
        self.names = list()
        self.cigar = list()
        self.r_st = list()
        self.r_end = list()
        self.q_st = list()
        self.q_end = list()
        self.section = list()
        self.strand = list()
    def read_sam(self,sam_file_path):
        with open(sam_file_path) as sam_h:
            for line in sam_h:
                if line.startswith('@'):
                    continue
                split_line = line.split()
                cigar = split_line[5]
                if cigar=="*":
                    continue
                if int(split_line[1])&0X10!=0:
                    self.strand.append(-1)
                else:
                    self.strand.append(1)
                self.names.append(split_line[0])
                self.cigar.append(split_line[5])
                self.r_st.append(int(split_line[3]))
                q_len = len(split_line[9])
                consume_count = sum(map(int,re.findall(r'(\d+)[M,D,N,=,X]',cigar)))
                self.r_end.append(self.r_st[-1]+consume_count-1)
                begin_s = re.findall(r'^(\d+)S',cigar)
                end_s = re.findall(r'(\d+)S$',cigar)
                if len(begin_s) == 0:    
                    self.q_st.append(0)
                else:
                    self.q_st.append(int(begin_s[0]))
                if len(end_s) == 0:
                    self.q_end.append(q_len+self.q_st[-1]-1)
                else:
                    self.q_end.append(q_len+self.q_st[-1]-int(end_s[0]))
                self.section.append(split_line[9])                     
                
    def mapping(self,query_path,ref_path):
        mapper = mp.Aligner(ref_path,preset="map-ont")
        for name,seq,qual in mp.fastx_read(query_path):
            for hit in mapper.map(seq):
                self.names.append(name)
                self.cigar.append(hit.cigar_str)
                self.r_st.append(hit.r_st)
                self.r_end.append(hit.r_en)
                self.q_st.append(hit.q_st)
                self.q_end.append(hit.q_en)
                self.section.append(hit.ctg)
                self.strand.append(hit.strand)
def run(args):
    global FLAGS
    FLAGS = args
#if __name__=="__main__":
parser=argparse.ArgumentParser(prog='chiron_label',description='Labelling dataset for Chiron.')
parser.add_argument('-i','--input',default="/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/output_SL_BS50/align.sam", help="Sam file alignment, containing the alignment between the merged fastq file to genome sequence")
args=parser.parse_args(sys.argv[1:])
aligns = alignments()

#aligns.read_sam(args.input)

ref = "/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/Reference/lambda.fasta"
query = "/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/output_SL_BS50/merge.fastq"
aligns.mapping(query,ref)