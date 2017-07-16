#!/bin/bash

#PBS -N raw_ws2
#PBS -S /bin/bash
#PBS -o ../../job_reports/
#PBS -e ../../job_reports/
#PBS -l nodes=1:ppn=1
#PBS -l mem=20G
#PBS -l walltime=20:00:00
#PBS -q normal

python /shares/compbio/Group-Yang/htteng/deepBNS/ctcbns/ctcbns/utils/raw4sheng.py --input_dir /shares/compbio/Group-Yang/htteng/deepBNS/data/Lambda_R9.4/signal_label
