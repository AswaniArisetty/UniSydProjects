#!/usr/bin/python3
import sys
import re
from collections import Counter
from string import punctuation

def get_tag_counts(tags_list,locality_words):
    #print (locality_words)
    loc_stop=[]
    for x in locality_words:
        loc_stop=loc_stop+x.split(" ")
        loc_stop.append(x.strip().replace(" ",""))
    #loc_stop.add((x.strip().replace(" ","") for x in locality_words))
    #print (locality_words)
    months=set(['january','february','march','april','may','june','july','august','september','october','november','december'])
    mon_short=(x[:4] for x in months)
    if tags_list!=['']:
        tags=[x for x in tags_list if x not in loc_stop if x not in months if x not in mon_short ]
        tag_counts=dict(Counter(tags).items())
    else:
        tag_counts='~'
    return tag_counts

def read_input(file):
    for line in file:
        yield line    
    
def assign_locality_tophotoid():
    last_locality=None
    last_placeid=None
    photo_count={}
    last_count=0
    count=0
    tags_list=''
    locality_words=[]
    data=read_input(sys.stdin)
    
    for line in data:
        words=line.split("\t")
        place_id,locality=words[0].split('#',1)
        locality=locality.strip()
        count=words[1].strip()
        
        if len(words)>2:
            if words[2]!='~':
                tags=words[2].strip()#.split()
            else:
                tags=''#[]        
        
        
        if not last_locality:
            last_placeid=place_id
            if locality !='~':
                last_count=0
            else :
                last_count =int(count)
                tags_list=tags                
            last_locality=locality

        elif last_placeid != place_id :
            if last_count >0 and last_locality !='~':
               # remove punctuation and replace : with _ to help in converting to a dictionary. Also remove years
               tags_list=tags_list.strip(punctuation).replace(":","_").lower() 
               tags_list=re.sub(r"\d{4}","",tags_list)
               locality_words=last_locality.strip(punctuation).lower().split(",")
               print ("{}#{}\t{}\t{}".format(last_placeid,last_locality,last_count,get_tag_counts(tags_list.split(),locality_words)))
            last_placeid=place_id
            
            if locality!='~':
               last_count=0
               tags_list=''
            else :
               try:
                   last_count=int(count)
                   tags_list=tags                   
               except ValueError:
                   last_count=0
                   tags_list=''
            last_locality=locality
        
        else:
            if locality =='~' and last_locality !='~':
                last_count +=int(count)
                tags_list=tags_list+' '+tags
            elif locality=='~' and last_locality=='~':
                last_count +=int(count)
                tags_list=tags_list+' '+tags
            elif locality !='~' :
                tags_list=tags_list.strip(punctuation).replace(":","_").lower()
                tags_list=re.sub(r"\d{4}","",tags_list)
                locality_words=locality.strip(punctuation).lower().split(",")
                print ("{}#{}\t{}\t{}".format(place_id,locality,last_count,get_tag_counts(tags_list.split(),locality_words)))
                last_count=0
                last_locality=locality
                tags_list=''

    if last_count >0 and last_locality !='~':
        tags_list=tags_list.strip(punctuation).replace(":","_").lower()
        tags_list=re.sub(r"\d{4}","",tags_list)
        locality_words=last_locality.strip(punctuation).lower().split(",")
        #print(locality_words,last_locality)
        print ("{}#{}\t{}\t{}".format(last_placeid,last_locality,last_count,get_tag_counts(tags_list.split(),locality_words)))
        last_placeid=place_id
        last_count=0              

if __name__=='__main__' :
     assign_locality_tophotoid()

