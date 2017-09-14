# Chiron
## A basecaller for Oxford Nanopore Technologies' sequencers
Using a deep learning CNN+RNN+CTC structure to establish end-to-end basecalling for the nanopore sequencer.  
Built with **TensorFlow** and python 2.7 by members of the Coin Group at the Institute for Molecular Bioscience (University of Queensland).

## Install
### Install using `pip` (recommended)
If you currently have TensorFlow installed on your system, we would advise you to create a virtual environment to install Chiron into, this way there is no clash of versions etc.

If you would like to do this, the best options would be [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/), the more user-friendly [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/install.html), or through [anaconda](https://docs.continuum.io/anaconda/install/). After installing one of these and activating the virtual environment you will be installing Chiron into, continue with the rest of the installation instructions as normal.

To install with `pip`:

```
pip install chiron  
```
This will install Chiron, the CPU-only distribution of TensorFlow (and it's dependencies), and [`h5py`](https://github.com/h5py/h5py) (required for reading in `.fast5` files).

**Note**: If you are after the GPU version, follow the steps in the following section.

### Install from GitHub
This is currently the best install method if you are wanting to run Chiron on in GPU mode (`pip install` version is coming).
```
git clone https://github.com/haotianteng/chiron.git
cd chiron
```
You will also need to install dependencies.

For CPU-version:
```
pip install tensorflow==1.0.1  
pip install h5py
```
For GPU-version(Nvidia GPU required):
```
pip install tensorflow-gpu==1.0.1  
pip install h5py
```

For alternate/detailed installation instructions for TensorFlow, see their [fantastic documentation](https://www.tensorflow.org/).

## Basecall
### If installed from `pip`:
An example call to Chiron to run basecalling is:  
```
chiron call -i <input_fast5_folder> -o <output_folder>

```

### If installed from GitHub:

All Chiron functionality can be run from **entry.py** in the Chiron folder. (You might like to also add the path to Chiron into your path for ease of running).

```
python chiron/entry.py call -i <input_fast5_folder> -o <output_folder>

```

### Test run

We provide 5 sample fast5 files (courtesy of [nanonet](https://github.com/nanoporetech/nanonet)) in the GitHub repository which you can run a test on. These are located in `chiron/example_data/`. From inside the Chiron repository:
```
python chiron/entry.py call -i chiron/example_folder/ -o <output_folder>
```


### Output
`chiron call` will create five folders in `<output_folder>` called `raw`, `result`, `segments`, `meta`, and `reference`.

* `result`: fastq/fasta files with the same name as the fast5 file they contain the basecalling result for. To create a single, merged version of these fasta files, try something like `paste --delimiter=\\n --serial result/*.fasta > merged.fasta` 
* `raw`: Contains a file for each fast5 file with it's raw signal. This file format is an list of integers. i.e `544 554 556 571 563 472 467 487 482 513 517 521 495 504 500 520 492 506 ... `
* `segments`: Contains the segments basecalled from each fast5 file.
* `meta`: Contains the meta information for each read (read length, basecalling rate etc.). Each file has the same name as it's fast5 file.
* `reference`: Contains the reference sequence (if any).

### Output format
With -e flag to output fastq file(default) with quality score or fasta file.  
Example:  
chiron call -i <input_fast5_folder> -o <output_folder> -e fastq  


chiron call -i <input_fast5_folder> -o <output_folder> -e fasta  

## Training
Usually the default model works fine on the R9.4 protocol, but if the basecalling result is not satisfying, you can train a model on your own training data set.  

#### Hardware request:  
Recommend training on GPU with TensorFlow - usually 8GB RAM (GPU) is required.  

#### Prepare the training data set.  
Using raw.py script to extract the signal and label from the re-squiggled fast5 file.
For how to re-squiggle fast5 file, check [here, nanoraw re-squiggle](https://nanoraw.readthedocs.io/en/latest/resquiggle.html#example-commands)
```
python chiron/utils/raw.py --input_dir <fast5 folder> --output_dir <output_folder>
```
`.signal` file and correspond `.label` file, a typical file format:  

`.signal` file format:  
`544 554 556 571 563 472 467 487 482 513 517 521 495 504 500 520 492 506 ...`  
i.e the file must contain only one row/column of raw signal numbers.  

`.label` file format:
```
70 174 A  
174 184 T  
184 192 A  
192 195 G  
195 204 C  
204 209 A  
209 224 C  
...  
```

Each line represents a DNA base pair in the Pore.
* 1st column: Start position of the current nucleotide, position related to the signal vector (index count starts from zero).  
* 2nd column: End position of the current nucleotide.  
* 3rd column: Nucleotide, for DNA: A, G, C, or T. Although, there is no reason you could not use other labels.

#### Adjust Chiron parameters
Go in to `chiron/chiron_rcnn_train.py` and change the hyper parameters in the `FLAGS` class.

```py
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
```

`data_dir`: The folder containing your signal and label files.  
`log_dir`: The folder where you want to save the model.  
`sequence_len`: The length of the segment you want to separate the sequence into. Longer length requires larger RAM.  
`batch_size`: The batch size.  
`step_rate`: Learning rate of the optimizer.  
`max_step`: Maximum step of the optimizer.  
`k_mer`: Chiron supports learning based on k-mer instead of a single nucleotide, this should be an odd number, even numbers will cause an error.  
`model_name`: The name of the model. The record will be stored in the directory `log_dir/model_name/`
`retrain`: If this is a new model, or you want to load the model you trained before. The model will be loaded from  `log_dir/model_name/`  

### Train!
```
source activate tensorflow   
python chiron/chiron_rcnn_train.py  
```
