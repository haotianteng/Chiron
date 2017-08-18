# chiron
## A basecaller for Nanopore Sequencer
Using a cnn+rnn+ctc structure to establish a end-to-end basecalling for the nanopore sequencer.  
Build with **Tensorflow** and python 2.7  

## Install
### Install using pip
$ pip install chiron  
This will install chiron and a CPU-only tensorflow for you.  
### Install from github
$ git clone https://github.com/haotianteng/chiron.git  



## Basecall
### If install from Pip:
The chiron command should be able to run direclty:  
$ chiron call -i <input_fast5_folder> -o <output_folder>  

### If install from github:
#### Before run  
Installing [Tensorflow by Google](https://www.tensorflow.org/).  
Using pip:
$ pip install tensorflow-gpu==1.0.1  
Or install CPU-only version  
$ pip install tensorflow==1.0.1  
Recommend to install tensorflow in a virtual environment, like **anaconda**.  
Then run the **entry.py** in the **chiron** folder:
$ python /path/to/chiron/entry.py -i <input_fast5_folder> -o <output_fast5_folder>

###Result:
This will create four folder called **raw**,**result**,**segments**,**meta** and **reference** in the output folder   
**result** folder: Fasta file with the same name of fast5 file containing the basecalling result.  
**raw** folder: Containing raw signal file.   
.signal file format:  
544 554 556 571 563 472 467 487 482 513 517 521 495 504 500 520 492 506 ...  
**segments** folder: Containing the segments basecalled from each fast5 file.
**meta** folder:Containing the meta information for each read. Same name as fast5 file.
**reference** folder: Containing the reference sequence.(If any)

## Training
Usually the default model works fine on the R9.7 protocal, but if the basecalling result is not satisfying, you can train your own model based on your own training data set.  

1. Hardware request:  
Recommand training on the GPU with tensorflow, usually a 8GB RAM is required.  

2. Prepare the training data set.  
Need .signal file and correspond .label file, a typical file format:  
  
.signal file format:  
544 554 556 571 563 472 467 487 482 513 517 521 495 504 500 520 492 506 ...  

One line of the signal or one column of the signal.  

.label file format:  
70 174 A  
174 184 T  
184 192 A  
192 195 G  
195 204 C  
204 209 A  
209 224 C  
...  

Each line represent a DNA base pair in the Pore: <start_position  end_position  nucleotide>  
1st column: Start position of the current nucleotide, position related to the signal vector, index count start with zero.  
2nd column: End position of the current nucleotide.  
3rd column: Nucleotide, for DNA is A G C T  

3. Go in to the chiron/chiron_rcnn_train.py and change the hyper parameters in FLAGS class:  
class Flags():  
    def __init__(self):  
        self.home_dir = "/home/haotianteng/UQ/deepBNS/"  
        self.data_dir = self.home_dir + 'data/Lambda_R9.4/raw/'  
        self.log_dir = self.home_dir+'/chiron/log/'  
        self.sequence_len = 200  
        self.batch_size = 100  
        self.step_rate = 1e-3   
        self.max_steps = 2500  
        self.k_mer = 1  
        self.model_name = 'crnn5+5_res_moving_norm'  
        self.retrain = False  
data_dir:the folder contained .signal and .label file.  
log_dir:the folder where you want to save the model.  
sequence_len: The length of the segment you want to separate the sequence into. Longer length require large RAM.  
batch_size: The batch size.  
step_rate:Learning rate of the optimizer.  
max_step: Maximum step of the optimizer.  
k_mer: chiron support learning based on k-mer instead of the single nucleotide, this should be an odd number, even number will cause error.  
model_name: the name of the model, the record will be stored in the directory log_dir/model_name/  
retrain: If this is a new model, or you want to load the model you trained before, also the model will be load from the directory log_dir/model_name/  

4. Train  
$source activate tensorflow   
$python chiron/chiron_rcnn_train.py  



