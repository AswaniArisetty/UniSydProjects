#!/bin/bash

if [ $# -ne 2 ]
then 
echo "usage: task1_driver.sh script_to_submit out_folder_name"
exit
fi

out_dir=${2}_task1
script=$1

hdfs dfs -rm -r $out_dir

spark-submit --master=yarn $script $out_dir

hadoop fs -getmerge $out_dir ./task1_output.txt 

echo "Task completed. Output available in ./task1_output.txt locally and in $out_dir on hdfs"
