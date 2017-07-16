#!/bin/bash
if [ $# -lt  1 ]
then
	echo "./extract_raw.sh <input_fast5_dir>"
	exit
fi
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "$DIR"
output="$(dirname "$1")/signal_label"
mkdir -p $output
#---- run oneline_signal_label.sh parser ---#
for file in $1/*.fast5
do
	filename=$(basename "$file")
	filename="${filename%.*}"
	bash Signal_Label_Package/oneline_signal_label.sh "$file" "${output}/${filename}"
done
rm "${output}/*.genome_label" "${output}/*.raw_dat" "${output}/*.raw_event" "${output}/*.raw_signal"
python raw4sheng.py --input_dir $output
