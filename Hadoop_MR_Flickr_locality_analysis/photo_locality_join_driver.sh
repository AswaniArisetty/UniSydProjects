#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./job_chain_driver.sh [place_file_location] [input_location] [output_location]"
    exit 1
fi

hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
-D stream.num.map.output.key.fields=2 \
-D stream.combiner.output.field.separator=# \
-D mapred.text.key.comparator.options="-k1,1 -k2r"  \
-D map.output.key.field.separator=# \
-D mapreduce.partition.keypartitioner.options=-k1,1 \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapreduce.job.maps=5 \
-D mapreduce.job.combines=5 \
-D mapreduce.job.reduces=15 \
-D mapreduce.job.name='Associate placeid with locality' \
-files photo_locality_join_reducer.py,photo_locality_join_combiner.py,photo_locality_join_mapper.py  \
-mapper photo_locality_join_mapper.py \
-combiner photo_locality_join_combiner.py \
-reducer photo_locality_join_reducer.py \
-input $1 \
-input $2 \
-output $3 \
-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner
