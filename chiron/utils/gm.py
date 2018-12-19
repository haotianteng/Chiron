import numpy as np
import json
import os
import sys
import argparse
from tqdm import tqdm
class gm:
    """
    Genome Model Class:
        k: the length of the longest kmer counted in the model, The genome model
        P(x|K) give the probability of observe Nucleotide x given a kmer K that has
        a length <= k.
    """
    def __init__(self,k = 5,mode = 0):
        self.k = k
        self.n = int(4*(4**k-1)/3) #n = 4**1 + 4**2 + ... + 4**k
        if mode == 0:
            self.base = ['A','C','G','T']
        elif mode == 1:
            self.base = ['A','C','G','U']
        self.kmer_dict = {}
        self.kmer_count = np.zeros([self.n,4],dtype = int)
        for i in range(self.n):
            kmer = self._idx2kmer(i)
            self.kmer_dict[kmer] = i
    def _kmer2idx(self,kmer):
        idx = 0
        for b_idx,b in enumerate(kmer):
            idx += (self.base.index(b)+1)*len(self.base)**b_idx
        idx = idx - 1
        return idx
    def _idx2kmer(self,idx):
        idx += 1
        kmer = ''
        while idx > 0:
            kmer = self.base[idx%len(self.base)-1] + kmer
            idx = int((idx-1) / len(self.base))
        return kmer
    def count_kmer(self,seq):
        for i in tqdm(range(len(seq)),desc = 'Counting kmer',position = 1):
            if i==0:
                pass
            for k in range(min(self.k,i)):
                kmer = seq[i-k-1:i]
                if self._base_check(kmer):
                    self.kmer_count[self.kmer_dict[kmer]][self.base.index(seq[i])] +=1
    def save(self,sav_path):
        gm_dict = self.__dict__
        gm_dict['kmer_count'] = gm_dict['kmer_count'].tolist()
        with open(sav_path, 'w+') as f:
            json.dump(gm_dict,f)
    def load(self,model_path):
        with open(model_path,'r') as f:
            gm_dict = json.load(f)
        self.k = gm_dict['k']
        self.n = gm_dict['n']
        assert self.n == int(4*(4**self.k-1)/3)
        self.base = gm_dict['base']
        self.kmer_dict = gm_dict['kmer_dict']
        self.kmer_count = np.asarray(gm_dict['kmer_count'])
    def _base_check(self,kmer):
        for base in kmer:
            if base not in self.base:
                return False
        return True
    def get_count(self,kmer):
        return self.kmer_count[self.kmer_dict[kmer]]
    def __getitem__(self, key):
        if type(key) is str:
            return self.get_count(key)
        elif (type(key) is int) or (type(key) is slice):
            return self.kmer_count[key]
        else:
            raise TypeError("Key shuold be a kmer string or int index.")
    def get_kmer_between(self,min_k,max_k):
        """
        Get the kmer whose length l, min_k <= l <= max_k
        """
        min_index = self._kmer2idx(self.base[0]*min_k)
        max_index = self._kmer2idx(self.base[-1]*max_k)
        return(min_index,max_index,self.kmer_count[min_index:max_index+1])
def fasta_reader(file_list,root_folder = None):
    for name in file_list:
        if root_folder is not None:
            name = os.path.join(root_folder,name)
        seqs = {}
        with open(name,'r') as f:
            for line in f:
                if line.startswith('>'):    
                    last_seq =line[1:].strip()
                    seqs[last_seq] = ''
                else:
                    seqs[last_seq]  = seqs[last_seq]+line.strip()
        yield name,seqs

def fastq_reader(file_list,root_folder = None):
    for name in file_list:
        if root_folder is not None:
            name = os.path.join(root_folder,name)
        seqs = {}
        with open(name,'r') as f:
            for line in f:
                if line.startswith('@'):    
                    last_seq =line[1:].strip()
                    seqs[last_seq] = ''
                    seq_line = next(f)
                    while not seq_line.startswith('+'):
                        seqs[last_seq]  = seqs[last_seq]+seq_line.strip()
                        seq_line = next(f)
        yield name,seqs
def run(args):
    root_folder = args.input
    f_L = os.listdir(root_folder)
    fasta_list = list()
    fastq_list = list()
    for f in f_L:
        if f.endswith('fasta') or f.endswith('.fa'):
            fasta_list.append(f)
        if f.endswith('fastq') or f.endswith('.fq'):
            fastq_list.append(f)
    gm1 = gm(k=args.k,mode = args.mode)
    if 'a' in args.suffixs:
        for genome,seqs in fasta_reader(fasta_list,root_folder):
            for name in tqdm(seqs.keys(),desc = "Reading genome "+genome,position = 0):
                gm1.count_kmer(seqs[name])
    if 'q' in args.suffixs:
        for genome,seqs in fastq_reader(fastq_list,root_folder):
            for name in tqdm(seqs.keys(),desc = "Reading genome "+genome,position = 0):
                gm1.count_kmer(seqs[name])        
    out_path = os.path.join(args.output,args.name)
    gm1.save(out_path)
    print("Genome model saved to %s"%(out_path))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='genome model',
                                     description='Build a genome model upon genomes.')
    parser.add_argument('-i', 
                        '--input', 
                        required = True,
                        help="Directory of the genome files.")
    parser.add_argument('-o',
                        '--output', 
                        required = True,
                        help="Output folder.")
    parser.add_argument('-s',
                        '--suffixs',
                        default = 'qa',
                        help="The suffix of the genome file, default is qa, a for only fasta, q for only fastq.")
    parser.add_argument('-n',
                        '--name',
                        default = "gm.json",
                        help="Output file name.")
    parser.add_argument('-k',
                        default = 6, 
                        type = int, 
                        help="The length of longest Kmer counted in the model")
    parser.add_argument('--mode',
                        type = int,
                        default = 0,
                        help = "Mode, if input is 0 (dna) or 1 (rna).")
    args = parser.parse_args(sys.argv[1:])
    run(args)
    

    