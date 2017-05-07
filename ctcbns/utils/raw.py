#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 04:16:40 2017

@author: haotianteng
"""
import labelop
import os,argparse,sys
parser = argparse.ArgumentParser(description='Transfer fast5 to raw_pair file.')
parser.add_argument('-i','--input_dir', help="Directory that store the fast5 files.")
args = parser.parse_args()
basecall_subgroup = 'BaseCalled_template'
basecall_group = 'Basecall_1D_000';
def extract(raw_folder = None):
    count = 1
    root_folder = args.input_dir
    if not os.path.isdir(root_folder):
        raise IOError('Input directory does not found.')
    if raw_folder is None:
        raw_folder = os.path.abspath(os.path.join(root_folder, os.pardir))+'/raw/'
    if not os.path.isdir(raw_folder):
        os.mkdir(raw_folder)
    for file_n in os.listdir(root_folder):
        if file_n.endswith('fast5'):
            output_file = raw_folder+os.path.splitext(file_n)[0]
            success = extract_file(root_folder+os.path.sep+file_n,output_file)
            if success:
                count +=1
            sys.stdout.write("%s file transfered.   \n" % (file_n) )
def extract_file(input_file,output_file):
    try:
        (raw_data, raw_label, raw_start, raw_length) = labelop.get_label_raw(input_file,basecall_group,basecall_subgroup)
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
def main():
    extract()
            
if __name__ == '__main__':
    root_folder = '/home/haotianteng/UQ/deepBNS/data/Lambda_R9.4'
    main()
