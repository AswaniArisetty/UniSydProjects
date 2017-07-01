#!/bin/bash

if [ $# -lt 3 ] || [ $# -gt 5 ]
then 
echo "usage: task2_driver.sh script_to_submit out_folder_name no_of_clusters[K] no_of_iterations dimensions"
echo "example: task2_driver.sh clusterCounts.py Assignment2 20 10 '11,6,7' "
exit
fi

out_dir=$2
task2_outdir=${out_dir}_task2
task3_outdir=${out_dir}_task3
num_executors=3
K=$3

if [ -z $4 ]
    then 
    no_of_iter=10
else 
    no_of_iter=$4
fi

if [ -z $5 ]
    then
    indices="11,6,7"
else 
    indices=$5
fi


script=$1

hdfs dfs -rm -r $task2_outdir $task3_outdir

spark-submit --master=yarn --num-executors=$num_executors $script $task2_outdir $task3_outdir $K $no_of_iter $indices


hadoop fs -getmerge $task2_outdir ./task2_output.txt 
hadoop fs -getmerge $task3_outdir ./task3_output.txt 


echo "Task completed. Output available in ./task2_output.txt and ./task3_output.txt locally "
echo "Also on HDFS at $task2_outdir , $task3_outdir"
