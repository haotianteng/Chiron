# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
#This Source Code Form is subject to the terms of the Mozilla Public
#License, v. 2.0. If a copy of the MPL was not distributed with this
#file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
#Created on Mon Aug 14 18:38:18 2017
import argparse
import sys
import logging
from os import path

from chiron import chiron_eval
from chiron import chiron_rcnn_train
from chiron.utils import raw
from chiron.utils.extract_sig_ref import extract


def evaluation(args):
    FLAGS = args
    FLAGS.input_dir = FLAGS.input
    FLAGS.output_dir = FLAGS.output
    FLAGS.recursive = True
    extract(FLAGS)
    FLAGS.input = FLAGS.output + '/raw/'
    chiron_eval.run(args)


def export(args):
    raw.run(args)


def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog='chiron', description='A deep neural network basecaller.')
    subparsers = parser.add_subparsers(title='sub command', help='sub command help')
    model_default_path = path.join(path.abspath(path.dirname(__file__)), 'model', 'DNA_default')
    print("model_default_path", model_default_path)
    # parser for 'call' command
    parser_call = subparsers.add_parser('call', description='Perform basecalling', help='Perform basecalling.')
    parser_call.add_argument('-i', '--input', required=True, help="File path or Folder path to the fast5 file.")
    parser_call.add_argument('-o', '--output', required=True, help="Output folder path")
    parser_call.add_argument('-m', '--model',type = str, default=model_default_path, help="model folder path")
    parser_call.add_argument('-s', '--start', type=int, default=0, help="Start index of the signal file.")
    parser_call.add_argument('-b', '--batch_size', type=int, default=1100,
                             help="Batch size for run, bigger batch_size will increase the processing speed but require larger RAM load")
    parser_call.add_argument('-l', '--segment_len', type=int, default=300, help="Segment length to be divided into.")
    parser_call.add_argument('-j', '--jump', type=int, default=30, help="Step size for segment")
    parser_call.add_argument('-t', '--threads', type=int, default=0,
                             help="Threads number, default is 0, which use all the available threads.")
    parser_call.add_argument('-e', '--extension', default='fastq', help="Output file type.")
    parser_call.add_argument('--beam', type=int, default=50,
                             help="Beam width used in beam search decoder, default is 50, set to 0 to use a greedy decoder. Large beam width give better decoding result but require longer decoding time.")
    parser_call.add_argument('--concise', action='store_true',
                             help="Concisely output the result, the meta and segments files will not be output.")
    parser_call.add_argument('--mode', default = 'dna',
                        help="Output mode, can be chosen from dna or rna.")
    parser_call.set_defaults(func=evaluation)

    # parser for 'extract' command
    parser_export = subparsers.add_parser('export', description='Export signal and label from the fast5 file.',
                                          help='Extract signal and label in the fast5 file.')
    parser_export.add_argument('-i', '--input', required=True, help='Input folder contain fast5 files.')
    parser_export.add_argument('-o', '--output', required=True, help='Output folder.')
    parser_export.add_argument('-f', '--tffile', default="train.tfrecords", help="tfrecord file")
    parser_export.add_argument('--basecall_group', default='Basecall_1D_000',
                        help='Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser_export.add_argument('--basecall_subgroup', default='BaseCalled_template',
                        help='Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    parser_export.set_defaults(func=export)

    # parser for 'train' command
    parser_train = subparsers.add_parser('train', description='Model training', help='Train a model.')
    parser_train.add_argument('-i', '--data_dir', required=True, help="Folder containing the labelled data.")
    parser_train.add_argument('-f', '--tfrecord', default="train.tfrecords", help='tfrecord file')
    parser_train.add_argument('-o', '--log_dir', required=True, help="Log dir which save the trained model")
    parser_train.add_argument('-m', '--model_name', required=True, help="Model name saved.")
    parser_train.add_argument('-c', '--cache_dir', default=None, help="Output folder")
    parser_train.add_argument('-t', '--retrain', type=bool, default=False,
                              help="If retrain is true, the previous trained model will be loaded from LOG_DIR before training.")
    parser_train.add_argument('-l', '--sequence_len', type=int, default=300, help="Segment length to be divided into.")
    parser_train.add_argument('-b', '--batch_size', type=int, default=400,
                              help="Batch size to train, large batch size require more ram but give faster training speed.")
    parser_train.add_argument('-x', '--max_steps', type=int, default=20000, help="Maximum training steps conducted.")
    parser_train.add_argument('-r', '--step_rate', type=float, default=1e-3,
                              help="Learning rate used for optimiztion algorithm.")
    parser_train.add_argument('-k', '--k_mer', type=int, default=1, help="Output k-mer size.")
    parser_train.set_defaults(func=chiron_rcnn_train.run)

    args = parser.parse_args(arguments)
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(sys.argv[1:])
    main()
