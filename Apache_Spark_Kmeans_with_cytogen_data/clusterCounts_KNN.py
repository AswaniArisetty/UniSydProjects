import math
import sys
import numpy as np
from pyspark import SparkContext

def find_closest_mean(kmeans, point):
    dist_array = []
    min_distance = float('inf')
    for i, mean_row in enumerate(kmeans):
        #print (i)
        distance = 0
        for k in range(len(mean_row)):
            distance += (point[k] - mean_row[k]) ** 2
        distance = math.sqrt(distance)
        dist_array.append(i)
        if distance < min_distance:
            min_distance = distance
            closest_mean_index = i
    return (closest_mean_index,(point,1,min_distance))
    #return dist_array


def find_closest_mean_np(kmeans,point):
    closest_mean_index=0
    min_distance=float('inf')
    for i in range(len(kmeans)):
        distance=np.sqrt(np.sum(point-kmeans[i]))
        if distance < min_distance:
            min_distance=distance
            closest_mean_index=i
    return (closest_mean_index,(point,1,min_distance))


def measurement_filter(line):
    cols = line.split(",")
    try:
        fsca = float(cols[1])
        ssca = float(cols[2])
        if fsca >= 1 and fsca <= 150000 and ssca >= 1 and ssca <= 150000:
            return True
        else:
            return False
    except ValueError:
        return False


def measurement_split(line, indices):
    y = line.split(",")
    # try:
    point_array = [float(y[i]) for i in indices]
    return point_array
    # except ValueError:
    #    print (y[i],"failed")
    #    point_array=['Error',y[i]]
    #    return point_array


def add_points(p1, p2):
    sum=0
    centroid_arr= [i1+i2 for i1,i2 in zip(p1[0],p2[0])]
    sum=p1[1]+p2[1]
    return([centroid_arr,sum])


def get_new_means(row):
    mean=[]
    for _,v in enumerate(row[1][0]):
        mean.append(v/row[1][1])
    return (row[0],mean,row[1][1])

    
def mcoll_10per(row):
    ind=row[0]
    total_num=int(len(row[1])*0.9)
    dist_filt=row[1][total_num]
    return (ind,dist_filt)



def usage():
    print ("usage :"+sys.argv[0]+" task2_output_dir task3_output_dir no_of_iters(K) dimension_indices" )
    print ("example: task2_driver.sh clusterCounts.py Assignment2 10 '4,6,7' ")
    return


    
def rem_10_per(row):
    count=len(row[1])//10
    after_outliers=sorted(row[1],reverse=True,key=lambda x: x[2])[count:]
    return (row[0],after_outliers)



    

if __name__ == '__main__':

    if len(sys.argv)!=6:
        usage()
        sys.exit()
    try: 
        task2_outdir=sys.argv[1]
        task3_outdir=sys.argv[2]
        K=int(sys.argv[3])
        iters=int(sys.argv[4])
        indices=map(int,sys.argv[5].split(","))
    except ValueError:
        print ("Exception processing arguments")
        usage()
        sys.exit()

    sc = SparkContext(appName="Assignment2_tasks_2_3_spark")
    measurements = sc.textFile("/share/cytometry/large/*.csv").filter( \
        measurement_filter).cache()
    experiments = sc.textFile("/share/cytometry/experiments.csv")
    measure_map = measurements.map(lambda row: measurement_split(row, indices)).cache()
    kmeans = measure_map.takeSample(False, K, 1)
    for i in range(iters):
         closest = measure_map.map(lambda row: find_closest_mean(kmeans, row))
         clusters = closest.reduceByKey(add_points).map(get_new_means).sortBy(lambda x: x[0])
         kmeans = [p[1] for p in clusters.collect()]
         #print (kmeans)
    clusters.map(lambda x: str(x[0])+'\t'+str(x[2])+'\t'+'\t'.join(map(str,x[1])) ).saveAsTextFile(task2_outdir)
    ##mcoll=measure_map.map(lambda row: find_closest_mean(kmeans, row))
    closest=measure_map.map(lambda row: find_closest_mean(kmeans, row))
    measures_filt=closest.groupByKey().map(rem_10_per).flatMap(lambda x: [(x[0],y) for y in x[1]]).cache()
    #mcoll_map=closest.filter(lambda x : x[1][0]<=dist_filters[x[0]][1]).cache()
    for i in range(iters):
        closest = measures_filt.map(lambda row: find_closest_mean(kmeans, row[1][0]))
        clusters = closest.reduceByKey(add_points).map(get_new_means).sortBy(lambda x: x[0])
        kmeans = [p[1] for p in clusters.collect()]
    #    #print (kmeans)
    
    clusters.map(lambda x: str(x[0])+'\t'+str(x[2])+'\t'+'\t'.join(map(str,x[1])) ).saveAsTextFile(task3_outdir) 

