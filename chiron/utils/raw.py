# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Mon Apr 10 04:16:40 2017

from __future__ import absolute_import
import argparse
import os
import sys

from chiron.utils import labelop
import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def extract(raw_folder=None):
    count = 1
    root_folder = FLAGS.input + os.path.sep
    output_folder = FLAGS.output + os.path.sep
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if output_folder is None:
        output_folder = os.path.abspath(os.path.join(root_folder, os.pardir)) + '/raw/'
    if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

    tfrecords_filename = output_folder + FLAGS.tffile

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for file_n in tf.gfile.ListDirectory(root_folder):
        if file_n.endswith('fast5'):
            output_file = output_folder + os.path.splitext(file_n)[0]
            success, (raw_data, raw_data_array) = extract_file(
                root_folder + os.path.sep + file_n)
            if success:
                count += 1
                example = tf.train.Example(features=tf.train.Features(feature={
                    'raw_data': _bytes_feature(raw_data.tostring()),
                    'features': _bytes_feature(raw_data_array.tostring()),
                    'fname':_bytes_feature(file_n)}))
                writer.write(example.SerializeToString())
            sys.stdout.write("%s file transfered.   \n" % (file_n))

    writer.close()


def extract_file(input_file):
    try:
        (raw_data, raw_label, raw_start, raw_length) = labelop.get_label_raw(
            input_file, FLAGS.basecall_group,
            FLAGS.basecall_subgroup)
    except IOError:
        return False, (None, None)
    except:
        return False, (None, None)

    raw_data_array = []
    for index, start in enumerate(raw_start):
        if raw_length[index]==0:
            print("input_file:" + input_file)
            raise ValueError("catch a label with length 0")
        raw_data_array.append(
            [start, start + raw_length[index], str(raw_label['base'][index])])
    if FLAGS.mode=='rna':
        raw_data = raw_data[::-1]
    return True, (raw_data, np.array(raw_data_array, dtype='S8'))


def run(args):
    global FLAGS
    FLAGS = args
    extract()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer fast5 to raw_pair file.')
    parser.add_argument('-i', '--input', required = True,
                        help="Directory that store the fast5 files.")
    parser.add_argument('-o', '--output', required = True, help="Output folder")
    parser.add_argument('-f', '--tffile', default="train.tfrecords",
                        help="tfrecord file")
    parser.add_argument('--basecall_group', default='RawGenomeCorrected_000',
                        help='Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser.add_argument('--mode',default = 'dna',
                        help='Type of data to basecall, default is dna, can be chosen from dna, rna and methylation(under construction)')
    args = parser.parse_args(sys.argv[1:])
    run(args)

