#!/usr/bin/python3
import sys
from pyspark import SparkContext 

def measurement_split(line):
    y=line.split(",")
    return (y[0],1)


def exp_split(line):
    y=line.split(",")
    ret_arr=[(y[0],x.strip()) for x in y[-1].split(";")]
    return ret_arr
    

def measurement_filter(line):
    cols=line.split(",")
    try:
        fsca=float(cols[1])
        ssca=float(cols[2])
        if fsca >=1 and fsca <=150000 and ssca >=1 and ssca <=150000:
            return True
        else :
            return False
    except ValueError:
        return False


def measure_count(prev_count,curr_count):
    return prev_count+curr_count


def usage():
    print ("usage: researcherCounts.py output_dir" )
    return


if __name__=='__main__':
    if len(sys.argv)!=2:
        usage()
        sys.exit() 
    outdir=sys.argv[1]
    sc = SparkContext(appName="Assignment2_task1_spark")
    measurements=sc.textFile("/share/cytometry/large/*.csv").cache()
    experiments=sc.textFile("/share/cytometry/experiments.csv")
    meas_map=measurements.filter(measurement_filter).map(measurement_split)
    exp_map=experiments.filter(lambda x: x.split(",")[0]!='sample').flatMap(exp_split)
    
    meas_exp_join=meas_map.join(exp_map)
    researcher_counts=meas_exp_join.map(lambda x: (x[1][1],x[1][0])).reduceByKey(measure_count,numPartitions=None).sortBy(lambda x: (-x[1],x[0]))
    output_counts=researcher_counts.map(lambda x: (x[0]+'\t'+str(x[1])))
    output_counts.saveAsTextFile(outdir)
    
    
    
    
    
    
