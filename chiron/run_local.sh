#!/bin/bash
source activate tensorflow
export PYTHONPATH=$PYTHONPATH:/home/haotianteng/UQ/deepBNS/
python chiron_rcnn_train.py
