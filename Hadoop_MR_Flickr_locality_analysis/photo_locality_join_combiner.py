#!/usr/bin/python3
import sys

def read_input(file):
    for line in file:
        yield line    

def combine_photo_count():
    """
    Combines photo counts of various place ids.
    Place ids need to be grouped 
    So needs partitioning,map key separator and order
    as specified in the driver script. 
    """
    last_locality=None
    last_placeid=None
    last_count=0
    count=0
    tags_list=''
    
    data=read_input(sys.stdin)
    
    for line in data:
        words=line.split("\t")
        place_id,locality=words[0].split('#',1)
        count=words[1].strip()
        if len(words)>2:
            if words[2]!='~':
                tags=words[2].strip()#.split()
            else:
                tags=''#[]
        
        # First line initializations
        
        if not last_locality:
            last_placeid=place_id
            if locality !='~':
                last_count=0
            else :
                last_count =int(count)
                tags_list=tags
            last_locality=locality
        
        #if transition occurs between place ids 
        #below block prints the output format
        #based on last_count
        # if last_count==0 row is from place.txt without any photos or tags
        # if last_count > 0 , row is from photos.txt with counts and tags
        # we do the join in reducer hence we print lines separately in combiner
        
        elif last_placeid != place_id :
             if last_count ==0:
                print ("{}#{}\t~\t~".format(last_placeid,last_locality))
                last_locality=locality
             elif last_count >0:
                print ("{}#{}\t{}\t{}".format(last_placeid,last_locality,last_count,tags_list))
             last_placeid=place_id
             last_locality=locality
             if locality!='~':
                 last_count=0
                 tags_list=''#[]
             else:
                 try:
                    last_count=int(count) 
                    tags_list=tags
                 except ValueError:
                    last_count=0
                    tags_list=''#[]
        
        # If the lines belong to same place id
        # below blocks print output based on the locality value
        # If locality=='~' row is from photo/n*.txt else from place.txt
        # we do the join in reducer hence we print lines separately in combiner
        
        else:
            if locality =='~' and last_locality !='~':
                print ("{}#{}\t~\t~".format(last_placeid,last_locality))
                last_count +=int(count)
                tags_list=tags_list+' '+tags
                last_locality=locality
            elif locality=='~' and last_locality=='~':
                last_count +=int(count)
                tags_list=tags_list+''+tags
            else :
                print ("{}#{}\t{}\t{}".format(last_placeid,last_locality,last_count,tags_list))
                last_count=0
                tags_list=''
                last_locality=locality
    
    # Handle the last row
    if last_count ==0:
        print ("{}#{}\t~\t~".format(last_placeid,last_locality))
        last_placeid=place_id
        last_count=0
        last_locality=locality
    elif last_count >0:
        print ("{}#{}\t{}\t{}".format(last_placeid,last_locality,last_count,tags_list))
        last_placeid=place_id
        last_count=0            
            
if __name__=="__main__":
    combine_photo_count()
