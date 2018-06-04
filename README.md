# Chiron
## A basecaller for Oxford Nanopore Technologies' sequencers
Using a deep learning CNN+RNN+CTC structure to establish end-to-end basecalling for the nanopore sequencer.  
Built with **TensorFlow** and python 2.7.

If you found Chiron useful, please consider to cite:  
> Teng, H., et al. (2017). Chiron: Translating nanopore raw signal directly into nucleotide sequence using deep learning. [bioRxiv 179531] (https://www.biorxiv.org/content/early/2017/09/12/179531)

---
## Table of contents

- [Install](#Install)
    - [Install using `pip`](#install-using-pip)
    - [Install from GitHub](#install-from-github)
- [Basecall](#basecall)
    - [If installed from `pip`](#if-installed-from-pip)
    - [If installed from GitHub](#if-installed-from-github)
    - [Test run](#test-run)
    - [Decoder choice](#decoder-choice)
    - [Output](#output)
    - [Output format](#output-format)
- [Training](#training)
    - [Hardware request](#hardware-request)
    - [Prepare training data set](#prepare-training-data-set)
    - [Train a model](#train-a-model)
    - [Training parameters](#training-parameters)
- [Train on Google Cloud ML engine](#train-on-google-cloud-ml-engine)
    - [Local testing](#local-testing)
    - [Create a new bucket](#create-a-new-bucket)
    - [Copy fast5 files to your Cloud Storage bucket](#copy-fsat5-files-to-your-cloud-storage-bucket)
    - [Transfer fast5 files to tfrecord](#transfer-fast5-files-to-tfrecord)
    - [Train model on Google Cloud ML engine](#train-model-on-google-cloud-ml-engine)
- [Distributed training on Google CLoud ML Engine](#distributed-training-on-google-cloud-ml-engine)
    - [Configure](#configure)
    - [Transfer fast5 files](#transfer-fast5-files)
    - [Submit training request](#submit-training-request)
## Install
### Install using `pip` (recommended)
If you currently have TensorFlow installed on your system, we would advise you to create a virtual environment to install Chiron into, this way there is no clash of versions etc.

If you would like to do this, the best options would be [`virtualenv`](https://virtualenv.pypa.io/en/stable/installation/), the more user-friendly [`virtualenvwrapper`](https://virtualenvwrapper.readthedocs.io/en/latest/install.html), or through [anaconda](https://docs.continuum.io/anaconda/install/). After installing one of these and activating the virtual environment you will be installing Chiron into, continue with the rest of the installation instructions as normal.

To install with `pip`:

```
pip install chiron  
```
This will install Chiron, and [`h5py`](https://github.com/h5py/h5py) (required for reading in `.fast5` files).
Tensorflow need to be install in addition by:
```  
pip install tensorflow  
```  
or GPU version:  
```  
pip install tensorflow-gpu  
```  

### Install from Source
```
git clone https://github.com/haotianteng/chiron.git
cd chiron
```
You will also need to install dependencies.
```
pip install h5py
pip install tqdm
pip install statsmodels
```
For CPU-version:
```
pip install tensorflow  
```
For GPU-version(Nvidia GPU required):
```
pip install tensorflow-gpu  
```
And then add the Chiron into PYTHONPATH,for convinience you can add it to the .bashrc
```
export PYTHONPATH=[Path to Chiron/Chiron]:$PYTHONPATH
```
For alternate/detailed installation instructions for TensorFlow, see the [documentation](https://www.tensorflow.org/).

## Basecall
### If installed from `pip`:
An example call to Chiron to run basecalling is:  
```
chiron call -i <input_fast5_folder> -o <output_folder> -m <model_folder>

```

### If installed from Github:

All Chiron functionality can be run from **entry.py** in the Chiron folder. (You might like to also add the path to Chiron into your PATH for ease of running).

```
python chiron/entry.py call -i <input_fast5_folder> -o <output_folder> -m <model_folder>

```

### Test run

We provide 5 sample fast5 files (courtesy of [nanonet](https://github.com/nanoporetech/nanonet)) in the GitHub repository and two models (DNA_default and RNA_default) which you can run a test on. These are located in `chiron/example_data/`. From inside the Chiron repository:
```
python chiron/entry.py call -i chiron/example_folder/ -o <output_folder> -m chiron/model/DNA_default
```

### Decoder choice
(From v0.3)  
Beam search decoder: chiron call -i <input> -o <output> --beam <beam_width>  
Greedy decoder: chiron call -i <input> -o <output> --beam 0  

Beam Seach decoder give a higher accuracy, and larger beam width can furthur improve the accuracy.
Greedy decoder give a faster decoding speed than the beam search decoder:  

| Device | Greedy decoder rate(bp/s) | Beam Search decoder rate(bp/s), beam_width=50 |  
| --- | --- | --- |  
| CPU | 21 | 17 |  
| GPU | 1652 | 1204 |  



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
The default DNA model trained on R9.4 protocol with a mix of Lambda and E.coli dataset, and the default RNA model is trained on R9.4 direct RNA kit (-200mV configuration).
If the basecalling result is not satisfying, you can train a model on your own training data set.  

#### Hardware request:  
Recommend training on GPU with TensorFlow - usually 8GB RAM (GPU) is required.  

#### Prepare training data set.  
Using raw.py script to extract the signal and label from the re-squiggled fast5 file.
(For how to re-squiggle fast5 file, check [here, nanoraw re-squiggle](https://nanoraw.readthedocs.io/en/latest/resquiggle.html#example-commands))

#### If installed from `pip`:
```
chiron export -i <fast5 folder> -o <output_folder>
```

or directly use the raw.py script in utils.

```
python chiron/utils/raw.py --input <fast5 folder> --output <output_folder> --mode dna
```

This will generate a tfrecord file for training when using the chiron_rcnn_train.py and chiron_input.py pipeline.  

```
python chiron/utils/file_batch.py --input <fast5 folder> --output <output folder> --length 400 --mode dna
```

This will generate several binary .bin file for training when using the chiron_train.py and chiron_queue_input.py pipeline.  

### Train a model

```
source activate tensorflow   
```
#### If installed from `pip`:
```
chiron train --data_dir <signal_label folder> --log_dir <model_log_folder> --model_name <saved_model_name>
```

or run directly by  

```
python chiron/chiron_rcnn_train.py  --data_dir <signal_label folder/ tfrecord file> --log_dir <model_log>
```
### Training parameters
Following parameters can be passed to Chiron when training

`data_dir`(Required): The folder containing your signal and label files.  
`log_dir`(Required): The folder where you want to save the model.  
`model_name`(Required): The name of the model. The record will be stored in the directory `log_dir/model_name/`
`tfrecord`: File name of tfrecord. Default is train.tfrecords.
`sequence_len`: The length of the segment you want to separate the sequence into. Longer length requires larger RAM.  
`batch_size`: The batch size.  
`step_rate`: Learning rate of the optimizer.  
`max_step`: Maximum step of the optimizer.  
`k_mer`: Chiron supports learning based on k-mer instead of a single nucleotide, this should be an odd number, even numbers will cause an error.  
`retrain`: If this is a new model, or you want to load the model you trained before. The model will be loaded from  `log_dir/model_name/`  

## Train on Google Cloud ML engine


### local testing

Before training the model on cloud ml engine, please check if it is working on local machine or not by following commands

```
gcloud ml-engine local train \
    --module-name chiron.utils.raw \
    --package-path chiron.utils/  \
    -- --input input_fast5_folder \
    --output output

gcloud ml-engine local train \
    --module-name chiron.chiron_rcnn_train \
    --package-path chiron/
```

If it is working well, please go to next step.

### create a new bucket

```
BUCKET_NAME=chiron-ml
REGION=us-central1
gsutil mb -l $REGION gs://$BUCKET_NAME
```

### Use gsutil to copy the all fast5 files to your Cloud Storage bucket.

```
gsutil cp -r raw_fast_folder gs://$BUCKET_NAME/fast5-data
```

### Train model on google cloud ML engine
```
JOB_NAME=chiron_single_1
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
INPUT_PATH=gs://$BUCKET_NAME/train_tfdata
```
```
gcloud ml-engine jobs submit training $JOB_NAME \
    --staging-bucket gs://chiron-ml \
    --module-name chiron.chiron_rcnn_train \
    --package-path chiron/ \
    --region $REGION \
    --config config.yaml \
    -- \
    --data_dir gs://$BUCKET_NAME/train_tfdata \
    --cache_dir gs://$BUCKET_NAME/cache/train.hdf5 \
    --log_dir gs://$BUCKET_NAME/GVM_model
```

## Distributed training on Google Cloud ML Engine

### Configure 
```
Change configure.yaml according to [GCloud Docs](https://cloud.google.com/ml-engine/docs/training-overview) 
For example the following configure_multi_gpu.yaml: 
 
trainingInput: 
  scaleTier: CUSTOM 
  masterType: standard_p100 
  workerType: standard_p100 
  parameterServerType: large_model 
  workerCount: 3 
  parameterServerCount: 3 
 
Will enable 3 workers + 1 master worker with one P-100 GPU in each worker. 
```

### Transfer fast5 files 
```
FAST5_FOLDER=/my/fast5/
OTUPUT_FOLDER=/my/file_batch/
SEGMENT_LEN=512
```

**Transfer fast5 to file batch**
```
python utils/file_batch.py --input $FAST5_FOLDER --output $OUTPUT_FOLDER --length $SEGMENT_LEN
```
**Copy to Google Cloud**
```
gsutil cp -r $OUTPUT_FOLDER gs://$BUCKET_NAME/file_batch
```
### Submit training request
```
JOB_NAME=chiron_multi_4
DATA_BUCKET=chiron-training-data
MODEL_BUCKET=chiron-model
REGION=us-central1
MODEL_NAME=test_model1
GPU_NUM=4
```
```
gcloud ml-engine jobs submit training ${JOB_NAME} \
    --runtime-version 1.6 \
    --staging-bucket gs://chiron-model/ \
    --module-name chiron.chiron_multi_gpu_train \
    --package-path chiron \
    --region $REGION \
    --config config_multi_gpu.yaml \
    -- \
    -i gs://$DATA_BUCKET/file_batch \
    -o gs://$MODEL_BUCKET/ \
    -m ${MODEL_NAME} \
    -n ${GPU_NUM}
```

