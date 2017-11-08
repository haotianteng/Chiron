#!/bin/bash
if [ $# -lt 2 ]
then
	echo -e "merge.sh <input_folder> <output_file> \nMerge the fasta file in the <input_folder> and output it as <output_file>"
	exit
fi
input_folder=$1
output_folder=$(dirname $2)
mkdir -p $output_folder
output=$2
> $output
a=1
for file_n in $input_folder/*.fasta;
do
	file_name=$(basename $file_n)
	sequence_name="${file_name%.*}"
	echo ">sequence${a} ${sequence_name}" >> $output
	cat $file_n | tail -1 >> $output
	echo >> $output
	a=$((a+1))
done