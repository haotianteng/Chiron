#!/bin/bash

#PBS -N crnn3+3_S10
#PBS -S /bin/bash
#PBS -o ../job_reports/
#PBS -e ../job_reports/
#PBS -l nodes=1:ppn=16
#PBS -l mem=120G
#PBS -l walltime=100:00:00
#PBS -q normal

cd $PBS_O_WORKDIR
sh delta_run_ctcbns_cpu.sh
