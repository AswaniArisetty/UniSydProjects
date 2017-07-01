#!/bin/bash

if [ $# -ne 2 ]
    then
    echo "usage:"
    echo "./all_tasks_driver.sh output_dir task1/2/3"
    exit
fi
outdir=$1
joindir=${outdir}_join
countdir=${outdir}_count
hdfs dfs -rm -r $outdir $joindir $countdir

task=$2

echo `date`
echo "Starting join job"
./photo_locality_join_driver.sh /share/place.txt /share/photo/n0?.txt $joindir 
echo " Join completed"

if [ $task == 'task1' ]
then
    ./count_photos_driver.sh $joindir $countdir  
    hadoop fs -getmerge $countdir ./assignment1_task1.txt
    echo "task1 completed output available at $countdir on hdfs, ./assignment1_task1.txt locally"
    exit
elif [ $task == 'task2' ]
then
    ./count_photos_driver.sh $joindir $countdir  
    ./sort_photo_counts_driver.sh $countdir $outdir 
    hadoop fs -getmerge $outdir ./top50_assignment1_task2.txt
    echo "task2 completed. output available at $outdir in hdfs and ./top50_assignment1_task2.txt locally "
    exit
elif [ $task == 'task3' ]
then
./count_photos_driver.sh $joindir $countdir 'with_tags'  
./sort_photo_counts_driver.sh $countdir $outdir 'with_tags' 
hadoop fs -getmerge $outdir ./top50_assignment1_task3.txt
echo "merged output file is at ./top50_assignment1_task3.txt locally as well as $outdir on hdfs "
fi
