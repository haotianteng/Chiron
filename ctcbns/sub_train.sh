#!/bin/bash

#PBS -N crnn3+3_S10
#PBS -S /bin/bash
#PBS -o ../job_reports/
#PBS -e ../job_reports/
#PBS -l nodes=1:ppn=8
#PBS -l ngpus=1
#PBS -l mem=120G
#PBS -l walltime=120:00:00
#PBS -q gpu

cd $PBS_O_WORKDIR
sh delta_run_ctcbns.sh
