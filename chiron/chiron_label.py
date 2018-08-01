import h5py
import os
import mappy
import re
import argparse
import numpy as np
import subprocess
import shlex
import sys
import tqdm
from multiprocessing import Pool
DATA_FORMAT = np.dtype([('raw','<i2'),
                        ('norm_raw','<f8'),
                        ('norm_trans','<f8'),
                        ('start','<i4'),
                        ('length','<i4'),
                        ('base','S1')])

def extract_fastq(input_f,ref_f,mode = 0):
    """
    Args:
        input_f: intput fast5 file handle
        ref_f: file name of the reference
        mode: 0-dna, 1-rna
    """
    with h5py.File(input_f,'r') as input_fh:
        raw_signal = list(input_fh['/Raw/Reads'].values())[0]['Signal'].value
        raw_seq = input_fh['/Analyses/Basecall_1D_000/BaseCalled_template/Fastq'].value
        ref = mappy.Aligner(ref_f)
        align = ref.map(raw_seq)
        ref = mappy.Aligner(ref_f,preset = "map-ont",best_n = 5)
        aligns = ref.map(raw_seq.split(b'\n')[1])
        maxmapq = -np.inf
        align = None
        for aln in aligns:
            if aln.mapq > maxmapq:
                maxmapq = aln.mapq
                align = aln
        if align is None:
            print("FAIL MAPPING "+input_f)
        if align.strand == -1:
            ref_seq = mappy.revcomp(ref.seq(align.ctg,start = align.r_st,end = align.r_en))
        else:
            ref_seq = ref.seq(align.ctg,start = align.r_st,end = align.r_en)
        if (mode == 1) or (mode == -1):
            raw_signal = raw_signal[::-1]
    if ref_seq is None:
        print(input_f)
        print(aligns)
    return raw_signal,raw_seq,ref_seq

def write_output(prefix,raw_signal,ref_seq):
    signal_fn = prefix+'.signal'
    ref_fn = prefix+'.ref'
    with open(signal_fn,'w+') as sig_f:
        sig_f.write('\n'.join([str(sig) for sig in raw_signal]))
    with open(ref_fn,'w+') as ref_f:
        ref_f.write(">"+ os.path.basename(prefix)+'\n')
        ref_f.write(ref_seq)
    input_command = " -i "+ref_fn+ ' -p '+ signal_fn + ' -o ' + prefix+'.aln'
    return input_command

def parse_cwDTW(f_path):
    """
    f_path: file path of the cwDTW output file.
    """
    segs = list()
    with open(f_path,'r') as out_f:
        for line in out_f:
            split_line = re.split('\s+\|*[diff:]*\s*',line.strip())
            segs.append(split_line)
    segs = np.array(segs)
    _,index = np.unique(segs[:,7],return_index = True)
    index = np.sort(index)
    segs = segs[index,:]
    output = list()
    for idx,seg in enumerate(segs[:-1]):
        current = seg[[0,4,5,2]].tolist()
        current.append(int(segs[idx+1,2])-int(seg[2]))
        current.append(seg[7][2])
        output.append(tuple(current))
    return np.array(output,dtype = DATA_FORMAT)
def write_back(fast5_f,aln_matrix,raw,ref):
    """
    Args:
        fast5_f: handle of the fast5 files.
        
    """
    with h5py.File(fast5_f,'a') as fast5_fh:
        data = np.asarray(aln_matrix,dtype = DATA_FORMAT)
        if '/Analyses/cwDTWCorrected_000' in fast5_fh:
            del fast5_fh['/Analyses/cwDTWCorrected_000']
            event_h = fast5_fh.create_dataset('/Analyses/cwDTWCorrected_000/BaseCalled_template/Events', shape = (len(data),),maxshape=(None,),dtype = DATA_FORMAT)
            fastq_h = fast5_fh.create_dataset('/Analyses/cwDTWCorrected_000/BaseCalled_template/Raw',shape = (),dtype = h5py.special_dtype(vlen=str))
            ref_h = fast5_fh.create_dataset('/Analyses/cwDTWCorrected_000/BaseCalled_template/Reference',shape = (),dtype = h5py.special_dtype(vlen=str))
        else:
            event_h = fast5_fh.create_dataset('/Analyses/cwDTWCorrected_000/BaseCalled_template/Events', shape = (len(data),),maxshape=(None,),dtype = DATA_FORMAT)
            fastq_h = fast5_fh.create_dataset('/Analyses/cwDTWCorrected_000/BaseCalled_template/Raw',shape = (),dtype = h5py.special_dtype(vlen=str))
            ref_h = fast5_fh.create_dataset('/Analyses/cwDTWCorrected_000/BaseCalled_template/Reference',shape = (),dtype = h5py.special_dtype(vlen=str))
        event_h[...] = data
        fastq_h[...] = raw
        ref_h[...] = ref
def label(filename):
    if filename.endswith("fast5"):
        abs_fast5 = os.path.join(args.input,filename)
        try:
            raw_signal,raw_seq,ref_seq = extract_fastq(abs_fast5,args.ref)
        except:
            return()
        prefix = os.path.join(args.saving,os.path.splitext(filename)[0])
        input_cmd = write_output(prefix,raw_signal,ref_seq)
        cmd = "./utils/cwDTW_nano " + input_cmd +' -R ' + str(args.mode)
        args_cmd = shlex.split(cmd)
        p = subprocess.Popen(args_cmd)
        p.wait()
        align_matrix = parse_cwDTW(prefix+'.aln')
        write_back(abs_fast5,align_matrix,raw_seq,ref_seq)
def run():
    pool = Pool(args.thread)
    filelist = os.listdir(args.input)
    for _ in pool.imap_unordered(label,filelist):
#    for _ in tqdm.tqdm(pool.imap_unordered(label,filelist),total = len(filelist)):
        pass
    pool.close()
    pool.join()        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='chiron',
                                     description='A deep neural network basecaller.')
    parser.add_argument('-i', '--input', required = True,
                        help="Directory of the fast5 files.")
    parser.add_argument('-r', '--ref', required = True,
                        help="Reference file name")
    parser.add_argument('-m','--mode',default = 0,
                        help="If RNA pore model is used, 0 for DNA pore model, 1 for 200mV RNA pore model, -1 for 180mV RNA pore model, DEFAULT is 0.")
    parser.add_argument('-s','--saving',
                        help="Temporary saving folder.")
    parser.add_argument('-t','--thread',default = 1,type = int,
                        help="Thread number.")
    args = parser.parse_args(sys.argv[1:])
    run()

    