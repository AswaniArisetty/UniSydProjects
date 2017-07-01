#!/usr/bin/python3
import sys
from collections import Counter

def read_input(file):
    for line in file:
        yield line

def get_dicts(tag_str):
    t=tag_str.strip('{}').split(",")
    tag_dict={}
    if len(t)>1:
        #try:
        z=[(x.split(":")) for x in t]
        tag_dict=dict([(x.strip().replace("'",""),int(y)) for x, y in z])
    return tag_dict


def count_photos_per_locality(flag=None):
    """
    Expects records grouped by locality 
    Locality and photo count must be separated by #
    """
    last_locality=None
    last_photo_count=0
    last_tag_freq=Counter()
    data=read_input(sys.stdin)
    for line in data:
        if not flag:
            words=line.split("#")
            locality=words[0].strip()
            count=int(words[1].strip())
        else: 
            words=line.split("\t")
            locality=words[0].split('#')[0].strip()
            count=int(words[0].split('#')[1].strip())
            tags=get_dicts(words[1].strip())  

        if not last_locality :
            last_locality=locality
            last_photo_count=count
            if flag:
                last_tag_freq=Counter(tags)

        elif locality==last_locality:
            last_photo_count +=count
            if flag:
                last_tag_freq.update(tags)
        else:
            if flag:
                print("{}\t{}\t{}".format(last_locality,last_photo_count,dict(last_tag_freq.most_common(10))))
                last_tag_freq=Counter(tags)
            else: 
                print("{}\t{}".format(last_locality,last_photo_count))
            last_locality=locality
            last_photo_count=count
    if flag:
            print("{}\t{}\t{}".format(last_locality,last_photo_count,dict(last_tag_freq.most_common(10))))
            last_tag_freq=Counter(tags)
    else: 
            print("{}\t{}".format(last_locality,last_photo_count))
if __name__=='__main__':
    if len(sys.argv)>1:
        flag=sys.argv[1]
        count_photos_per_locality(flag)
    else:
        count_photos_per_locality()
