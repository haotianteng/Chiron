# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Temporary script.TODO: Integrate tombo or cwDTW for auto generating data.
from __future__ import absolute_import
from __future__ import print_function
import argparse
import os
import re
import sys

import h5py
import mappy as mp

from .chiron_input import biglist
from six.moves import map


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

    def read_sam(self, sam_file_path):
        with open(sam_file_path) as sam_h:
            for line in sam_h:
                if line.startswith('@'):
                    continue
                split_line = line.split()
                cigar = split_line[5]
                if cigar == "*":
                    continue
                if int(split_line[1]) & 0X10 != 0:
                    self.strand.append(-1)
                else:
                    self.strand.append(1)
                self.names.append(split_line[0])
                self.cigar.append(split_line[5])
                self.r_st.append(int(split_line[3]))
                q_len = len(split_line[9])
                consume_count = sum(map(int, re.findall(r'(\d+)[M,D,N,=,X]', cigar)))
                self.r_end.append(self.r_st[-1] + consume_count - 1)
                begin_s = re.findall(r'^(\d+)S', cigar)
                end_s = re.findall(r'(\d+)S$', cigar)
                if len(begin_s) == 0:
                    self.q_st.append(0)
                else:
                    self.q_st.append(int(begin_s[0]))
                if len(end_s) == 0:
                    self.q_end.append(q_len + self.q_st[-1] - 1)
                else:
                    self.q_end.append(q_len + self.q_st[-1] - int(end_s[0]))
                self.section.append(split_line[9])

    def mapping(self, query_path, ref_path):
        if os.path.isdir(query_path):
            file_list = os.listdir(query_path)
        else:
            file_list = [query_path]
        for file_name in file_list:
            mapper = mp.Aligner(ref_path, preset="map-ont")
            for name, seq, qual in mp.fastx_read(file_name):
                for hit in mapper.map(seq):
                    self.names.append(name)
                    self.cigar.append(hit.cigar_str)
                    self.r_st.append(hit.r_st)
                    self.r_end.append(hit.r_en)
                    self.q_st.append(hit.q_st)
                    self.q_end.append(hit.q_en)
                    self.section.append(hit.ctg)
                    self.strand.append(hit.strand)


def get_label(ref, align, read_index, signals, lens, segments):
    """
    Input Args:
        ref: reference sequence for the segment, data type: string
        align: An alignments class, the alignments for all the reads
        read_index: The index of the read in the align class.
        signals: list of the signal arrays. [segment_number,segment_length]
        lens: list of the length of the above signals. [segment_number]
        segments: list of the segment output. [segment_number,segment_sequence]
    """
    return None


def label():
    align = alignments()
    reference = dict()
    with open(FLAGS.reference) as ref_h:
        for line in ref_h:
            if line.startswith('>'):
                chrome = line.split()[0][1:]
                reference[chrome] = next(ref_h)

    if FLAGS.alignment is None:
        align.mapping(FLAGS.reads, FLAGS.reference)
    elif FLAGS.fastx is not None:
        align.read_sam(FLAGS.alignment)
    else:
        raise ValueError("Either a sam file or a output file need to be givien.")

    with h5py.File(FLAGS.output, "a") as hdf5_record:
        event_h = hdf5_record.create_dataset('event/record', dtype='float32', shape=(0, seq_length),
                                             maxshape=(None, seq_length))
        event_length_h = hdf5_record.create_dataset('event/length', dtype='int32', shape=(0,), maxshape=(None,),
                                                    chunks=True)
        label_h = hdf5_record.create_dataset('label/record', dtype='int32', shape=(0, 0), maxshape=(None, seq_length))
        label_length_h = hdf5_record.create_dataset('label/length', dtype='int32', shape=(0,), maxshape=(None,))
        event = biglist(data_handle=event_h, max_len=0)
        event_length = biglist(data_handle=event_length_h, max_len=0)
        label = biglist(data_handle=label_h, max_len=0)
        label_length = biglist(data_handle=label_length_h, max_len=0)
        for name in os.listdir(FLAGS.segments):
            read_index = None
            lens = list()
            signals = list()
            segments = list()
            if name.endswith('.fastq') or name.endswith('.fasta'):
                strand_name = os.path.splitext(name)[0]
                try:
                    read_index = align.names.index(strand_name)
                except ValueError:
                    print(strand_name + " is not find int the alignments.\n")
                    continue
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    continue
                with open(FLAGS.segments + os.sep + name) as seg_h:
                    for line in seg_h:
                        if line.startswith('>') or line.startswith('@'):
                            split_line = line.split('|')
                            infos = split_line[-1].split()
                            for info in infos:
                                key_pair = info.split('=')
                                if key_pair[0] == 'length':
                                    lens.append(int(key_pair[1]))
                                if key_pair[0] == 'signal':
                                    signals.append(list(map(float, key_pair[1].split(','))))
                            segments.append(next(seg_h))
            if read_index is not None:
                ref = reference[align.section[read_index]][align.r_st[read_index]:align.r_end[read_index]]
                tmp_event, tmp_event_length, tmp_label, tmp_label_length = get_label(ref, align, signals, lens,
                                                                                     segments)
                event += tmp_event
                event_length += tmp_event_length
                label += tmp_label
                label_length += tmp_label_length
                del tmp_event
                del tmp_event_length
                del tmp_label
                del tmp_label_length


def run(args):
    global FLAGS
    FLAGS = args


# if __name__=="__main__":
parser = argparse.ArgumentParser(prog='chiron_label', description='Labelling dataset for Chiron.')
parser.add_argument('-s', '--segments', default='/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/segments',
                    help="File folder of the segment files", required=True)
parser.add_argument('-r', '--reference', default='/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/Reference/lambda.fasta',
                    help="Reference genome.", required=True)
parser.add_argument('-o', '--output', help="The path of the output hdf5 file.", required=True)
parser.add_argument('-a', '--alignment', default=None,
                    help="Sam file alignment, containing the alignment between the merged fastq file to genome sequence")
parser.add_argument('-f', '--fastx', default=None,
                    help="The output fastq or fasta files, if a sam file is given, this input will be ignored")
parser.add_argument('-m', '--max', default=1000000, help="Maximum number of reads that labelled.")
args = parser.parse_args(sys.argv[1:])
aligns = alignments()
run(args)

# aligns.read_sam(args.input)

ref = "/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/Reference/lambda.fasta"
query = "/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4/output_SL_BS50/merge.fastq"
aligns.mapping(query, ref)
