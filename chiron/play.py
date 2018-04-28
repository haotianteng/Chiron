from chiron.utils.easy_assembler import simple_assembly
from pprint import pprint
import numpy as np
import edlib

start = 0
jump = 30
segment_len = 300

def index2base(read):
    """Transfer the number into dna base.
    The transfer will go through each element of the input int vector.
    Args:
        read (Int): An Iterable item containing element of [0,1,2,3].

    Returns:
        bpread (Char): A String containing translated dna base sequence.
    """

    base = ['A', 'C', 'G', 'T']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread

folder = "/tmp/ch2"
ext = "fasta"
read = "read1"
with open(f"{folder}/segments/{read}.{ext}") as f:
    read_seqments = [x.strip() for x in f.readlines()[1::2] if len(x)]

# pprint(read_seqments)
assembly, lb = simple_assembly(read_seqments)
assembly = np.argmax(assembly, axis=0)
lb = lb[assembly, np.arange(lb.shape[1])]
complete_assembly = index2base(assembly)
for i, x in enumerate(lb):
    print(x, complete_assembly[i])

with open(f"{folder}/result/{read}.{ext}") as f:
    from_ref = f.readlines()[1].strip()
pprint(edlib.align(complete_assembly, from_ref, task="path"))

# with open(f"{folder}/raw/{read}.signal") as f:
#     pprint(f.readlines())
