import h5py
import os
import mappy
import re
import argparse
import numpy as np
import subprocess
import shlex
import sys
from multiprocessing import Pool
DATA_FORMAT = np.dtype([('raw','<i2'),
                        ('norm_raw','<f8'),
                        ('norm_trans','<f8'),
                        ('start','<i4'),
                        ('length','<i4'),
                        ('base','S1')]) 
BASECALL_ENTRY = '/Analyses/Basecall_1D_000'
    
def extract_fastq(input_f,ref_f,mode = 0,trans_start = None):
    """
    Args:
        input_f: intput fast5 file handle
        ref_f: file name of the reference
        mode: 0-dna, 1-rna, -1-rna 180mV
        trans_start: Start position of the transcription(required in RNA mode).
    """
    with h5py.File(input_f,'r') as input_fh:
        raw_entry = list(input_fh['/Raw/Reads'].values())[0]
        raw_signal = raw_entry['Signal'].value
        raw_seq = input_fh[BASECALL_ENTRY+'/BaseCalled_template/Fastq'].value
        if mode !=0:
            assert trans_start is not None
            raw_signal,raw_seq = _decap(input_fh,trans_start,raw_signal,raw_seq)
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
    _,index = np.unique(segs[:,3],return_index = True)
    index = np.sort(index)
    segs = segs[index,:]
    output = list()
    for idx,seg in enumerate(segs[:-1]):
        current = seg[[0,4,5]].tolist()
        current.append(int(seg[2])-1)
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
        event_h.attrs['read_start_rel_to_raw'] = 0
        fastq_h[...] = raw
        ref_h[...] = ref
def read_tsv(tsv_file):
    """
    Read a Nanopolish TSV file.
    Args:
        tsv_file: nanopolish polya segmentation tsv file.
    """
    POLYA_FORMAT = [('position','<u8'),
                ('leader_start','<i4'),
                ('polya_start','<i4'),
                ('transcript_start','<i4'),
                ('read_rate','<f4'),
                ('polya_length','<f4'),
                ('qc_tag','a16')]
    polya_seg= dict()
    with open(tsv_file) as f:
        headline = f.readline().strip().split()
        for entry in headline:
            polya_seg[entry] = list()
        for line in f:
            split_line = line.strip().split()
            for idx,item in enumerate(split_line):
                polya_seg[headline[idx]].append(item)
    readnames = np.array(polya_seg['readname'])
    contigs = np.array(polya_seg['contig'])
    POLYA_FORMAT = [('readname',readnames.dtype),('contig',contigs.dtype)] + POLYA_FORMAT
    return np.array(list(zip(readnames,
                             contigs,
                             polya_seg['position'],
                             polya_seg['leader_start'],
                             polya_seg['adapter_start'],
                             polya_seg['polya_start'],
                             polya_seg['transcript_start'],
                             polya_seg['read_rate'],
                             polya_seg['polya_length'],
                             polya_seg['qc_tag'])),dtype = POLYA_FORMAT)
def read_link(readdb_f):
    """
    Read the link from the readdb_file
    Args:
        readdb_f:File path of the readdb file generated by nanopolish.
    """
    dest_link = dict()
    with open(readdb_f) as f:
        for line in f:
            split_l = line.strip().split()
            dest_link[split_l[0]] = split_l[0]
    return dest_link

def fast5s_iter(dest_link,tsv_table):
    """
    An iterator iterate over the fast5 files.
    Args:
        dest_link: readname -> fast5 path
        tsv_table: Polya segmentation information.
    Return:
        An interator of (start_position, transcript start position)
    """
    for idx,tag in enumerate(tsv_table['qc_tag']):
        if tag == "PASS":
            yield dest_link[tsv_table['readname'][idx]],tsv_table['transcript_start']
    
def _decap(fast5_root, trans_start,raw_signal,raw_seq):
    """
    Get the raw signal(RNA) with polya tail removed.
    Args:
        fast5_root: Root entry of the fast5 file.
        trans_start: start position of the transcription.
    """
    event_entry = fast5_root[BASECALL_ENTRY+'/BaseCalled_template/Events'].value
    moves = event_entry['move']
    skip_n = sum(moves[event_entry['start']<trans_start])
    return raw_signal[trans_start:],raw_seq[:-skip_n]
def label(abs_fast5):
    trans_start = abs_fast5[1]
    abs_fast5 = abs_fast5[0]
    if abs_fast5.endswith("fast5"):
        filename = os.path.basename(abs_fast5)                
        try:
            raw_signal,raw_seq,ref_seq = extract_fastq(abs_fast5,args.ref,args.mode,trans_start)
        except:
            return()
        prefix = os.path.join(args.saving,'cwdtw_output',os.path.splitext(filename)[0])
        fast5_svae = os.path.join(args.saving,'fast5s',filename)
        os.shutil.copyfile(abs_fast5,fast5_save)
        input_cmd = write_output(prefix,raw_signal,ref_seq)
        cmd = os.path.dirname(os.path.realpath(__file__))+"/utils/cwDTW_nano " + input_cmd +' -R ' + str(args.mode)
        args_cmd = shlex.split(cmd)
        p = subprocess.Popen(args_cmd)
        p.wait()
        align_matrix = parse_cwDTW(prefix+'.aln')
        write_back(fast5_save,align_matrix,raw_seq,ref_seq)

def run():
    if not os.path.isdir(args.saving):
        os.mkdir(args.saving)
    pool = Pool(args.thread)
    filelist = []
    for path , _ , files in os.walk(args.input):
        for file in files:
            if file.endswith('fast5'):
                filelist.append((os.path.join(path,file),None))
    for _ in pool.imap_unordered(label,filelist):
#    for _ in tqdm.tqdm(pool.imap_unordered(label,filelist),total = len(filelist)):
        pass
    pool.close()
    pool.join()        
def run_rna():
    if not os.path.isdir(args.saving):
        os.mkdir(args.saving)
    pool = Pool(args.thread)
    filelist = []
    for file,trans_start in fast5s_iter(args.dest_link,args.tsv_table):
        if file.endswith('fast5'):
            filelist.append((file,trans_start))
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
    parser.add_argument('-r', '--ref', required = True,extract_fastq
                        help="Reference file name")
    parser.add_argument('--polya',default = None,
                        help="Polya segment TSV file generated by Nanopolish, required in RNA mode.")
    parser.add_argument('--readdb',default = None,
                        help="Path of readdb file generated by Nanopolish, required in RNA mode")
    parser.add_argument('--index',default = None,
                        help="Readdb file index the sequence id and fast5 file path, required when resquiggle in RNA mode.")
    parser.add_argument('-m','--mode',default = 0,type = int,
                        help="If RNA pore model is used, 0 for DNA pore model, 1 for 200mV RNA pore model, -1 for 180mV RNA pore model, DEFAULT is 0.")
    parser.add_argument('-s','--saving',
                        help="Temporary saving folder.")
    parser.add_argument('-t','--thread',default = 1,type = int,
                        help="Thread number.")
    args = parser.parse_args(sys.argv[1:])
    if args.mode==1 or args.mode == -1 :
        run_rna()
    elif args.mode ==0 :
        run()
    else:
        raise ValueError("Mode should be 0,1 or -1")

    
