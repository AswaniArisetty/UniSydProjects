#!/usr/bin/python3
import sys

def read_input(file):
    for line in file:
        yield line

def as_is_out(flag=None):
    data = read_input(sys.stdin)
    for line in data:
        if flag:
            words=line.split("\t")
            locality=words[0].strip()
            count=words[1].strip()
            tags=words[2].strip()
            print("{}#{}\t{}".format(locality,count,tags))
        else :
            words=line.split("\t")
            locality=words[0].strip()
            count=words[1].strip()
            print("{}#{}".format(locality,count))


if __name__=='__main__':
    if len(sys.argv)>1:
        flag=sys.argv[1]
        as_is_out(flag)
    else:
        as_is_out()
