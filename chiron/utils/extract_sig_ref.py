# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Thu May  4 10:57:35 2017

import argparse
import os
import sys
import h5py
import logging
import threading
from tqdm import tqdm

def extract(FLAGS):
    # logger = logging.getLogger(__name__)
    tqdm.monitor_interval = 0
    count = 1
    root_folder = FLAGS.input_dir
    out_folder = FLAGS.output_dir
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if out_folder is None:
        raw_folder = os.path.abspath(os.path.join(out_folder, 'raw'))
        ref_folder = os.path.abspath(os.path.join(out_folder, 'reference'))
    else:
        if not os.path.isdir(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        raw_folder = os.path.abspath(os.path.join(out_folder, 'raw'))
        ref_folder = os.path.abspath(os.path.join(out_folder, 'reference'))
    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    if not os.path.isdir(ref_folder):
        os.mkdir(ref_folder)
    if FLAGS.recursive:
        dir_list = os.walk(root_folder)
    else:
        dir_list = [root_folder]
    for dir_tuple in tqdm(dir_list,desc = "Subdirectory processing:",position = 0):
        if FLAGS.recursive:
            directory = dir_tuple[0]
            file_list = dir_tuple[2]
        else:
            file_list = os.listdir(dir_tuple)
        for file_n in tqdm(file_list,desc = "Signal processing:",position = 1):
            if FLAGS.recursive:
                full_file_n = os.path.join(directory,file_n)
                # print(file_n)
            else:
                full_file_n = os.path.join(root_folder,file_n)
            if file_n.endswith('fast5'):
                try:
                    raw_signal, reference = extract_file(full_file_n,FLAGS.mode)
                    count += 1
                    if len(raw_signal) == 0:
                        raise ValueError("Failed in extracting " + (
                            os.path.join(raw_folder, os.path.splitext(file_n)[0] + '.signal')))
                except:
                    # logging.getLogger(__name__).error("Cannot extact file %s", file_n, exc_info=True)
                    continue
                with open(os.path.join(raw_folder, os.path.splitext(file_n)[0] + '.signal'), 'w+') as signal_file:
                    signal_file.write(" ".join([str(val) for val in raw_signal]))
                if len(reference) > 0:
                    with open(os.path.join(ref_folder, os.path.splitext(file_n)[0] + '_ref.fasta'), 'w+') as ref_file:
                        ref_file.write(reference)
                if (FLAGS.test_number is not None) and (count >=FLAGS.test_number):
                    return

def extract_file(input_file,mode = 'dna'):
    try:
        input_data = h5py.File(input_file, 'r')
    except IOError:
        return False
    except:
        return False
    raw_signal = list(input_data['/Raw/Reads'].values())[0]['Signal'].value
    if mode == 'rna':
        raw_signal = raw_signal[::-1]
    try:
        reference = input_data['Analyses/Basecall_1D_000/BaseCalled_template/Fastq'].value
        reference = '>template\n' + reference.split('\n')[1]
    except:
        try:
            reference = input_data['Analyses/Alignment_000/Aligned_template/Fasta'].value
        except:
            reference = ''
    return raw_signal, reference


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract the signal and reference from fast5 file.')
    parser.add_argument('-i', 
                        '--input_dir', 
                        required = True,
                        help="Directory that store the fast5 files.")
    parser.add_argument('-o', 
                        '--output_dir', 
                        required = True,
                        help="Directory that output the signal and reference sequence.")
    parser.add_argument('-r',
                        '--recursive',
                        action='store_true',
                        help="If recursively search subfolder")
    parser.add_argument('-m',
                        '--mode',
                        default = 'rna',
                        help="Output mode, can be chosen from dna or rna.")
    parser.add_argument('--test_number',
                        default = None,
                        type = int,
                        help="Extract test_number reads, default is None, extract all reads.")
    args = parser.parse_args(sys.argv[1:])
    extract(args)
