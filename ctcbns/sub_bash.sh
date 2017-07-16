#!/bin/bash
scriptname=sub_train.sh
if [ -z $5 ]; then
        name=$(basename $scriptname)
else
        name=$5
fi

mkdir -p ../job_reports

# $1 - command
# $2 - how many CPUs
# $3 - how much memory
# $4 - time
# $5 - job name
echo '#!/bin/bash' > $scriptname
echo ""            >> $scriptname
echo "#PBS -N ${name}" >> $scriptname
echo "#PBS -S /bin/bash" >> $scriptname
echo "#PBS -o ../job_reports/" >> $scriptname
echo "#PBS -e ../job_reports/" >> $scriptname
echo "#PBS -l nodes=1:ppn=$2" >> $scriptname
echo "#PBS -l ngpus=1" >> $scriptname
echo "#PBS -l mem=$3" >> $scriptname
echo "#PBS -l walltime=$4:00:00" >> $scriptname
echo "#PBS -q gpu" >> $scriptname
echo "" >> $scriptname
echo "cd \$PBS_O_WORKDIR" >> $scriptname
echo $1 >> $scriptname

qsub $scriptname
echo $scriptname
