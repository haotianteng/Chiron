#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 04:16:40 2017
Modified by Lee Yam Keng on Sat Feb 24 2018
@author: haotianteng, Lee Yam Keng
"""
import argparse
import os
import sys

import labelop
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

    tfrecords_filename = output_folder + 'train.tfrecords'

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for file_n in os.listdir(root_folder):
        if file_n.endswith('fast5'):
            output_file = output_folder + os.path.sep + os.path.splitext(file_n)[0]
            success, (raw_data, raw_data_array) = extract_file(root_folder + os.path.sep + file_n, output_file)
            if success:
                count += 1                
                example = tf.train.Example(features=tf.train.Features(feature={
                    'raw_data': _bytes_feature(raw_data.tostring()),
                    'features': _bytes_feature(raw_data_array.tostring())}))
                writer.write(example.SerializeToString())

            sys.stdout.write("%s file transfered.   \n" % (file_n))

    writer.close()


def extract_file(input_file, output_file):
    try:
        (raw_data, raw_label, raw_start, raw_length) = labelop.get_label_raw(input_file, FLAGS.basecall_group,
                                                                             FLAGS.basecall_subgroup)
    except IOError:
        return False, (None, None)
    except:
        return False, (None, None)

    f_signal = open(output_file + '.signal', 'w+')
    f_label = open(output_file + '.label', 'w+')
    f_signal.write(" ".join(str(val) for val in raw_data))
    raw_data_array = []
    for index, start in enumerate(raw_start):
        f_label.write("%d %d %c\n" % (start, start + raw_length[index], str(raw_label['base'][index])))
        raw_data_array.append([start, start + raw_length[index], str(raw_label['base'][index])])
    f_signal.close()
    f_label.close()

    return True, (raw_data, np.array(raw_data_array, dtype='S5'))


def run(args):
    global FLAGS
    FLAGS = args
    extract()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer fast5 to raw_pair file.')
    parser.add_argument('-i', '--input', help="Directory that store the fast5 files.")
    parser.add_argument('-o', '--output', default=None, help="Output folder")
    parser.add_argument('--basecall_group', default='Basecall_1D_000',
                        help='Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    args = parser.parse_args(sys.argv[1:])
    run(args)
