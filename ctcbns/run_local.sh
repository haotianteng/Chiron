#!/bin/bash
source activate tensorflow
export PYTHONPATH=$PYTHONPATH:/home/haotianteng/UQ/deepBNS/
python ctcbns_rcnn_train.py
