#!/bin/bash
home_dir=/shares/coin/haotian.teng/
export PYTHONPATH="$PYTHONPATH:${home_dir}/deepBNS/ctcbns/"
module load tensorflow/1.0.1.gpu
export LD_LIBRARY_PATH=${home_dir}/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=${home_dir}/cuda/include:$CPATH
export LIBRARY_PATH=${home_dir}/cuda/lib64:$LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=2
python ctcbns_rcnn_train.py
