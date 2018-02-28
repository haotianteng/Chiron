#!/bin/bash
if [ $# -lt 3 ]
then
	echo "assess <fasta_file> <reference_file> <output_result>"
	exit
fi
input=$1
ref=$2
output=$3
file_n="${output%.*}"
output="${file_n}.sam"
echo $file_n
echo $output
bash_dir="$(dirname "${BASH_SOURCE[0]}")"
${bash_dir}/graphmap align -r $ref -d $input -o $output
samtools view -bS $output > "${file_n}.bam"
>"${file_n}.aln"
jsa.hts.errorAnalysis --bamFile="${file_n}.bam" --reference=$ref > "${file_n}.aln"
