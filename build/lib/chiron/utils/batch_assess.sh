#!/bin/bash
if [ $# -lt 2 ]
then
	echo "batch_assess <output_folder> <Reference.fasta>"
	exit
fi
folder=$1
ref=$2
bash_dir="$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "${folder}/assess"
mkdir -p "${folder}/assess_metrichor"
for file_n in ${folder}/result/*.fasta
do
	file_name=$(basename $file_n)
	ext_name="${file_name%.*}"
	$bash_dir/assess.sh "${file_n}" $ref "${folder}/assess/${ext_name}.sam"
done

for file_n in ${folder}/reference/*.fasta
do
	file_name=$(basename $file_n)
	ext_name="${file_name%.*}"
	$bash_dir/assess.sh "${file_n}" $ref "${folder}/assess_metrichor/${ext_name}.sam"
done
