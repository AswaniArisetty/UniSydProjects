#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Invalid number of parameters!"
    echo "Usage: ./job_chain_driver.sh [place_file_location] [input_location] [output_location]"
    exit 1
fi

hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \
-D stream.num.map.output.key.fields=2 \
-D mapred.text.key.comparator.options="-k2nr -k1r"  \
-D map.output.key.field.separator=# \
-D mapred.output.key.comparator.class=org.apache.hadoop.mapred.lib.KeyFieldBasedComparator \
-D mapreduce.job.maps=5 \
-D mapreduce.job.reduces=1 \
-D mapreduce.job.name='sort localities on photo counts' \
-files sort_photo_counts_mapper.py,sort_photo_counts_reducer.py \
-mapper 'sort_photo_counts_mapper.py with_tags' \
-reducer 'sort_photo_counts_reducer.py with_tags' \
-input $1 \
-output $2 \

