#!/usr/bin/python3
import sys
from collections import Counter

def read_input(file):
    for line in file:
        yield line

def get_dicts(tag_str):
    #print (tag_str)
    t=tag_str.strip('{}').split(",")
    tag_dict={}
    if len(t)>1:
        #try:
        z=[(x.split(":")) for x in t]
        tag_dict=dict([(str(x).replace("'","").strip(),int(y)) for x, y in z])
        tag_dict=sorted(tag_dict.items()
                         ,key=lambda x:-x[1])
    return tag_dict


def filter_top_n(n=50,flag=None):
    i=1
    last_count=None
    data=read_input(sys.stdin)
    for line in data:
        if i>=n:
            continue
        if flag:
            words=line.split("\t")
            locality=words[0].split('#')[0].strip()
            count=int(words[0].split('#')[1].strip())
            tags=get_dicts(words[1].strip())
            print ("{}\t{}\t{}".format(locality,count,str(tags).strip("[]")))
        else:
            locality,count=line.strip().split("#")
            print ("{}\t{}".format(locality,count))
        if not last_count:
            last_count=count
        elif last_count!=count:
            i+=1
            


if __name__=='__main__':
    if len(sys.argv)>1:
       flag=sys.argv[1]
       filter_top_n(50,flag)
    else:
       filter_top_n(50)
