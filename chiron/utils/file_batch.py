# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Apr 10 04:16:40 2017
#Transfer fast5 into data batch.
#Need Python>3
import argparse
import os
import struct
import sys

import numpy as np
from statsmodels import robust

import labelop

DNA_BASE = {'A': 0, 'C': 1, 'G': 2, 'T': 3, }
DNA_IDX = ['A', 'C', 'G', 'T']


def extract():
    root_folder = FLAGS.input + os.path.sep
    output_folder = FLAGS.output + os.path.sep
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if output_folder is None:
        output_folder = os.path.abspath(os.path.join(root_folder, os.pardir)) + '/batch_/' + str(FLAGS.length)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    batch_idx = 1
    output_file = None
    event = list()
    event_length = list()
    label = list()
    label_length = list()
    success_list = list()
    fail_list = list()
    format_string = '<1H' + str(FLAGS.length) + 'f1H' + str(FLAGS.length) + 'b'

    def extract_fast5(input_file_path, bin_h, mode='DNA'):
        """
        Extract the signal and label from a single fast5 file
        Args:
            input_file_path: path of a fast5 file.
            bin_h: handle of the binary file.
            mode: The signal type dealed with. Default to 'DNA'.
        """
        try:
            (raw_data, raw_label, raw_start, raw_length) = labelop.get_label_raw(input_file_path, FLAGS.basecall_group,
                                                                                 FLAGS.basecall_subgroup)
        except IOError:
            fail_list.append(input_file_path)
            return False
        except:
            fail_list.append(input_file_path)
            return False
        if mode=='rna':
            print(type(raw_data))
            raw_data = raw_data[::-1]
        if FLAGS.normalization == 'mean':
            raw_data = (raw_data - np.median(raw_data)) / np.float(np.std(raw_data))
        elif FLAGS.normalization == 'median':
            raw_data = (raw_data - np.median(raw_data)) / np.float(robust.mad(raw_data))
        pre_start = raw_start[0]
        pre_index = 0
        for index, start in enumerate(raw_start):
            if start - pre_start > FLAGS.length:
                if index - 1 == pre_index:
                    # If a single segment is longer than the maximum singal length, skip it.
                    pre_start = start
                    pre_index = index
                    continue
                event.append(np.pad(raw_data[pre_start:raw_start[index - 1]],
                                    (0, FLAGS.length + pre_start - raw_start[index - 1]), mode='constant'))
                event_length.append(int(raw_start[index - 1] - pre_start))
                label_ind = raw_label['base'][pre_index:(index - 1)]
                temp_label = [DNA_BASE[x.decode('UTF-8')] for x in label_ind]
                label.append(
                    np.pad(temp_label, (0, FLAGS.length - index + 1 + pre_index), mode='constant', constant_values=-1))
                label_length.append(index - 1 - pre_index)
                pre_index = index - 1
                pre_start = raw_start[index - 1]
            if raw_start[index] - pre_start > FLAGS.length:
                # Skip a single event segment longer than the required signal length
                pre_index = index
                pre_start = raw_start[index]
        success_list.append(input_file_path)
        while len(event) > FLAGS.batch:
            for index in range(0, FLAGS.batch):
                bin_h.write(struct.pack(format_string,
                                        *[event_length[index]] + event[index].tolist() + [label_length[index]] + label[
                                            index].tolist()))
            del event[:FLAGS.batch]
            del event_length[:FLAGS.batch]
            del label[:FLAGS.batch]
            del label_length[:FLAGS.batch]
            return True
        return False

    for file_n in os.listdir(root_folder):
        if file_n.endswith('fast5'):
            if output_file is None:
                output_file = open(output_folder + os.path.sep + "data_batch_" + str(batch_idx) + '.bin', 'wb+')
            output_state = extract_fast5(root_folder + os.path.sep + file_n, output_file)
            if output_state:
                batch_idx += 1
                output_file.close()
                if (FLAGS.max is not None) and (batch_idx > FLAGS.max):
                    sys.stdout.write("Reach the maximum %d batch number, finish read." % (FLAGS.max))
                    break
                output_file = open(output_folder + os.path.sep + "data_batch_" + str(batch_idx) + '.bin', 'wb+')
                sys.stdout.write("%d batch transferred completed.\n" % (batch_idx - 1))
    sys.stdout.write("File batch transfer completed, %d batches have been processed\n" % (batch_idx - 1))
    sys.stdout.write("%d files scussesfully read, %d files failed.\n" % (len(success_list), len(fail_list)))
    if not output_state:
        output_file.close()
        os.remove(output_folder + os.path.sep + "data_batch_" + str(batch_idx) + '.bin')
    with open(output_folder + os.path.sep + "data.meta", 'w+') as meta_file:
        meta_file.write("signal_length " + str(FLAGS.length) + "\n")
        meta_file.write("file_batch_size " + str(FLAGS.batch) + "\n")
        meta_file.write("normalization " + FLAGS.normalization + "\n")
        meta_file.write("basecall_group " + FLAGS.basecall_group + "\n")
        meta_file.write("basecall_subgroup" + FLAGS.basecall_subgroup + "\n")
        meta_file.write("DNA_base A-0 C-1 G-2 T-3" + "\n")
        meta_file.write("data_type " + FLAGS.mode + "\n")
        meta_file.write("format " + format_string + "\n")
    return


def run(args):
    global FLAGS
    FLAGS = args
    extract()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer fast5 to file batch.')
    parser.add_argument('-i', '--input', required = True,
                        help="Directory that store the fast5 files.")
    parser.add_argument('-o', '--output', required = True,
                        help="Output folder")
    parser.add_argument('--basecall_group', default='RawGenomeCorrected_000',
                        help='Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser.add_argument('-l', '--length', default=512, help="Length of the signal segment")
    parser.add_argument('-b', '--batch', default=10000, help="Number of record in one file.")
    parser.add_argument('-n', '--normalization', default='median',
                        help="The method of normalization applied to signal, Median(default):robust median normalization, 'mean': mean normalization, 'None': no normalizaion")
    parser.add_argument('-m', '--max', default=10, help="Maximum number of batch files generated.")
    parser.add_argument('--mode', default='dna', help="Sequecing data type. Default is DNA.")
    args = parser.parse_args(sys.argv[1:])
    run(args)
