#!/usr/bin/python3
import sys
import ast
from collections import Counter
from string import punctuation

def read_input(file):
    for line in file:
        yield line

def map_counts_per_locality(flag=None):
    
    data=read_input(sys.stdin)
    
    for line in data:
        words=line.split("\t")
        photo_count=int(words[1].strip())
        locality=words[0].split('#')[1]
        tag_counts=words[2].strip()
        if photo_count >0:
            if flag:
                print ("{}#{}\t{}".format(locality,photo_count,tag_counts))
            else:
                print ("{}#{}".format(locality,photo_count))

if __name__=="__main__":
    if len(sys.argv)>1:
        flag=sys.argv[1]
        map_counts_per_locality(flag)
    else:
        map_counts_per_locality()
