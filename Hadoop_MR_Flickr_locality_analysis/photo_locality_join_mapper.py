#!/usr/bin/python3

import sys
from collections import Counter

def read_input(file):
    # Split each line into words
    for line in file:
        yield line

def multi_mapper():
    """ This mapper will output different format dependind on input type
    Input format: place_id \t woeid \t latitude \t longitude \t place_name \t place_type_id \t place_url OR
                  photo_id \t owner \t tags \t date_taken \t place_id \t accuracy
    Output format: place_id#locality \t ~ OR
                   place_id#~ \t 1 \t tags
    """
    data=read_input(sys.stdin)
    
    for line in data:
        # Clean input and split it
        parts = line.strip().split("\t")

        # Check that the line is of the correct format
        if len(parts) == 7:  # The line comes from place.txt
            place_id=parts[0]
            if parts[5] in ['22','7']:
                place_url_parts=parts[4].split(',')
                if parts[5]=='22':
                    locality=",".join(place_url_parts[1:]) # if 22 first filed is neighbourhood
                elif parts[5]=='7':
                    locality=",".join(place_url_parts)
                print("{}#{}\t~".format(place_id,locality) )
        elif len(parts) == 6:  # The line comes n0*.txt
            place_id = parts[4].strip()
            tags = parts[2].strip()
            #print(place_id + "#" +"-"+"\t"+ '1' + "\t" + tags)
            print ("{}#~\t1\t{}".format(place_id,tags))

if __name__ == "__main__":
    multi_mapper()
