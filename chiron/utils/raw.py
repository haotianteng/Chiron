#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 04:16:40 2017

@author: haotianteng
"""
import labelop
import os,argparse,sys

def extract(raw_folder = None):
    count = 1
    root_folder = FLAGS.input+os.path.sep
    output_folder = FLAGS.output+os.path.sep
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if output_folder is None:
        output_folder = os.path.abspath(os.path.join(root_folder, os.pardir))+'/raw/'
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    for file_n in os.listdir(root_folder):
        if file_n.endswith('fast5'):
            output_file = output_folder+os.path.splitext(file_n)[0]
            success = extract_file(root_folder+os.path.sep+file_n,output_file)
            if success:
                count +=1
            sys.stdout.write("%s file transfered.   \n" % (file_n) )
def extract_file(input_file,output_file):
    try:
        (raw_data, raw_label, raw_start, raw_length) = labelop.get_label_raw(input_file,FLAGS.basecall_group,FLAGS.basecall_subgroup)
    except IOError:
        return False
    except:
        return False
    f_signal = open(output_file+'.signal','w+')
    f_label = open(output_file+'.label','w+')
    f_signal.write(" ".join(str(val) for val in raw_data))
    for index,start in enumerate(raw_start):    
        f_label.write("%d %d %c\n"%(start,start+raw_length[index],str(raw_label['base'][index])))
    f_signal.close()
    f_label.close()
    return True
def run(args):
    global FLAGS
    FLAGS = args
    extract()
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Transfer fast5 to raw_pair file.')
    parser.add_argument('-i','--input', help="Directory that store the fast5 files.")
    parser.add_argument('-o','--output',default = None,help = "Output folder")
    parser.add_argument('--basecall_group',default = 'Basecall_1D_000',help = 'Basecall group Nanoraw resquiggle into. Default is Basecall_1D_000')
    parser.add_argument('--basecall_subgroup',default = 'BaseCalled_template',help = 'Basecall subgroup Nanoraw resquiggle into. Default is BaseCalled_template')
    args=parser.parse_args(sys.argv[1:])
    run(args)