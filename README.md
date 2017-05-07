# Basecaller for Nanopore Sequencer(BNS)
Using a cnn+rnn+ctc structure to establish a end-to-end basecalling for the nanopore sequencer.  
Build with **Tensorflow** and python 2.7  

## Before run  
Installing [Tensorflow by Google](https://www.tensorflow.org/).  
Recommend to install tensorflow in a virtual environment, like **anaconda**.  

## Run      
1. Transfer the .fast5 file into a .signal file.  
$python ctcbns/utils/extract_sig_ref.py --input_dir <fast5_file or directory> --output_dir <output directory>  

This will create two folder called raw and reference in the same folder of fast5_file_dir if the output_dir is not given, for example  
path_to_fast5_dir/fast5_file_dir  
then the raw folder will be created in   
path_to_fast5_dir/raw      	#folder contained the raw signal files
path_to_fast5_dir/reference	#folder contained the original base calling by **Metrichor**

If the output_dir is given, then the raw and reference folder will be generated under the output_dir
  
Format of .signal file:  
One line of the raw signal, like:  
544 554 556 571 563 472 467 487 482 513 517 521 495 504 500 520 492 506 ...  
Or one column of the raw signal:  
544  
554  
556  
571  
564  
472  
467  
487  
482  
513  
517  
...  

3.Base call!  
$python ctcbns/ctcbns_eval.py --input <.signal file or folder containing .signal file> --output <output_directory>  

## Train the model  
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

3. Go in to the ctcbns/ctcbns_rcnn_train.py and change the hyper parameters in FLAGS class:  
class Flags():  
    def __init__(self):  
        self.home_dir = "/home/haotianteng/UQ/deepBNS/"  
        self.data_dir = self.home_dir + 'data/Lambda_R9.4/raw/'  
        self.log_dir = self.home_dir+'/ctcbns/log/'  
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
k_mer: ctcbns support learning based on k-mer instead of the single nucleotide, this should be an odd number, even number will cause error.  
model_name: the name of the model, the record will be stored in the directory log_dir/model_name/  
retrain: If this is a new model, or you want to load the model you trained before, also the model will be load from the directory log_dir/model_name/  

4. Train  
$source activate tensorflow   
$python ctcbns/ctcbns_rcnn_train.py  


