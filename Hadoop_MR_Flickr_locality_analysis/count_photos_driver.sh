#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./job_chain_driver.sh [place_file_location] [input_location] [output_location]"
    exit 1
fi

hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
-D stream.num.map.output.key.fields=2 \
-D stream.combiner.output.field.separator=# \
-D mapred.text.key.comparator.options="-k1"  \
-D map.output.key.field.separator=# \
-D mapreduce.partition.keypartitioner.options=-k1,1 \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapreduce.job.maps=5 \
-D mapreduce.job.reduces=5 \
-D mapreduce.job.name='Associate placeid with locality' \
-files count_photos_mapper.py,count_photos_reducer.py  \
-mapper 'count_photos_mapper.py with_tags' \
-reducer 'count_photos_reducer.py with_tags' \
-input $1 \
-output $2 \
-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner
