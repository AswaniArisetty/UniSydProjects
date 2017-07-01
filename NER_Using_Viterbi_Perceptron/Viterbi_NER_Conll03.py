#!/usr/bin/python3 

import sys
import os
import random
import re
from collections import defaultdict,Counter
import numpy as np
from Perceptron_official import Instance,Perceptron
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix


def get_dicts(feature_array,lang):
    features=feature_array.split(" ")
    if lang=='eng':
        word=features[0]
        pos_tag=features[1]
        chunk_tag=features[2]
    elif lang=='deu':
        word=features[0]
        lemma=features[1]
        pos_tag=features[2]
        chunk_tag=features[3]
    return ({'word':word,'pos_tag':pos_tag,'chunk_tag':chunk_tag})

def get_feat_per_word(word,feat_prefix):
    feat_arr=[]
    feat_arr+=[word]
    #feat_arr+=[word[:4]]#,word[:3],word[:2]]
    feat_arr += [word[-4:]]#, word[-3:], word[-2:]]

    #feat_arr+=[lemma.lemmatize(word)]

    if wordcounts[word]<=5:
        #elif  re.match('\w*\W+.', '.ABC'):
        #    feat_arr+=[feat_prefix+'has_punctuation']
        if re.match('\W+$', word):
            feat_arr += ['punctuation_only']
            # elif  re.match('\w*\W+.', '.ABC'):
            #    feat_arr+=[feat_prefix+'has_punctuation']
        if re.match('\w*[a-z][A-Z]+\w*', word):
            feat_arr += ['mixedcase']
        if re.match('[0-9]+$', word):
            feat_arr += ['all_numbers']
        if word == word.upper():
            feat_arr += ['ALLCAPS']
        if word[0] == word[0].upper():
            feat_arr += ['initcap']
        if re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
            feat_arr += ['has_hyphen']
        if re.match('\w*\.', word):
            feat_arr += ['has_period']
        if re.match('\w+\.\w+', word):
            feat_arr += ['is_abbreviation']


    return feat_arr

def get_word_shape(word):
    word_shp = re.sub(r'[A-Z]', 'X', word)
    word_shp = re.sub(r'[a-z]', 'x', word_shp)
    word_shp = re.sub(r'[0-9]', '1', word_shp)
    return word_shp

def get_short_word_shp(word):
    short_word_shp = re.sub(r'[A-Z]+', 'X', word)
    short_word_shp = re.sub(r'[a-z]+', 'x', short_word_shp)
    short_word_shp = re.sub(r'[0-9]+', '1', short_word_shp)
    return  short_word_shp


def get_features(i,token,prev_token,prev2_token,next_token,next2_token,words):
    lemma=WordNetLemmatizer()
    features=[]#defaultdict(int)
    word,pos,chunk=token['word'],token['pos_tag'],token['chunk_tag']
    prev_word, prev_pos, prev_chunk = prev_token['word'], prev_token['pos_tag'], prev_token['chunk_tag']
    prev2_word, prev2_pos, prev2_chunk = prev2_token['word'], prev2_token['pos_tag'], prev2_token['chunk_tag']
    next_word, next_pos, next_chunk = next_token['word'], next_token['pos_tag'], next_token['chunk_tag']
    next2_word, next2_pos, next2_chunk = next2_token['word'], next2_token['pos_tag'], next2_token['chunk_tag']

    features+=get_feat_per_word(word,'word_')
    features.append(prev_word)
    features.append(prev2_word)
    features.append(next_word)
    features.append(next2_word)
    features.append('word_pos_' + pos)
    features.append('word_chunk_tag_'+chunk)
    features.append('prev_word_pos_' + prev_pos)
    features.append('prev_word_chunk_tag_' + prev_chunk)
    features.append('prev2_word_pos_' + prev2_pos)
    #features.append('prev2_word_chunk_tag_' + prev2_chunk)
    features.append('next_word_pos_' + next_pos)
    features.append('next_word_chunk_tag_' + next_chunk)
    features.append('next2_word_pos_' + next2_pos)
    #features.append('next2_word_chunk_tag_' + next2_chunk)
    features.append('word_lem' + lemma.lemmatize(word))
    features.append('prev_word_lem_'+lemma.lemmatize(prev_word))
    features.append('next_word_lem_'+lemma.lemmatize(next_word))

    short_word_shp=get_short_word_shp(word)
    short_prev_shp=get_short_word_shp(prev_word)
    short_next_shp=get_short_word_shp(next_word)
    features.append(short_prev_shp+' '+short_word_shp)
    features.append(short_word_shp + ' ' + short_next_shp)
    features.append(short_prev_shp+' '+short_word_shp + ' ' + short_next_shp)

    if word in loc_gaz:
         features.append('word_in_gaz')
    if prev_word in loc_gaz:
         features.append('prev_word_in_gaz')
    if next_word in loc_gaz:
         features.append('next_word_in_gaz')
    if word in loc_gaz and next_word in loc_gaz:
        features.append('word_and_next_word_in_gaz')
    if prev_word in loc_gaz and word in loc_gaz:
        features.append('prev_word_and_word_in_gaz')



    return features

def viterbi(sentence_feats,words):
    tagset=set(['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'O'])
    out_tag_seq=[]
    N=len(sentence_feats)+4
    #print (N)
    viterbi_chart={}
    perc_calls=0
    for i in range(N):
        viterbi_chart[i]={}

    viterbi_chart[0]['-START2-']=(0.0,'')
    viterbi_chart[1]['-START1-']=(0.0,'-START2-')
    prev_klass='-START1-'
    for i in range(2,N-2):
        #print (prev_klass)
        found_tag=False
        #print (words[i-2])
        word=words[i-2]
        tag_stored=tagdict.get(word)
        #print (tag_stored)
        if 1==2:#tag_stored:
            #print ("before change",prev_klass)
            viterbi_chart[i][tag_stored]=(100,prev_klass)
            prev_klass=tag_stored

            #print ("prev_klass",prev_klass)
            #print ("tag found in tag_dict")
        else:
            perc_calls += 1
            for tag in tagset:
                #for feats in sentence_feats:
                #perc_calls+=1
                score=perc._score(Instance(sentence_feats[i-2]+['prev_klass_'+prev_klass],''),tag)
                    #print (feats[0])
                #print (tag,weight)
                prev_list=[]
                for prev_tag in viterbi_chart[i-1]:
                    (prev_score,pre_prev_tag)=viterbi_chart[i-1][prev_tag]
                    prev_list.append((score+prev_score,prev_tag))
                #print ('prev_list:',prev_list)
                (best_score,best_prev_tag)=max(prev_list,key=lambda x: x[0])
                if best_score !=0.0:
                    viterbi_chart[i][tag]=(best_score,best_prev_tag)
                    found_tag=True
                    #print ('best_tag:',best_weight,tag,best_prev_tag)
            if found_tag is False:
                viterbi_chart[i]['O']=(0.0,'O')
                prev_klass='O'
            else :
                score=None
                best_tag=''
                for tag in viterbi_chart[i].keys():
                    x=viterbi_chart[i][tag]
                    if score == None:
                        score = x[0]
                        best_tag = tag
                    elif score < x[0]:
                        score = x[0]
                        best_tag = tag
                prev_klass=best_tag


    # find max value for last tag
    score=None
    best_tag=''
    for tag in viterbi_chart[N-3].keys():
        x=viterbi_chart[N-3][tag]
        if score==None:
            score=x[0]
            best_tag=tag
        elif score < x[0]:
            score=x[0]
            best_tag=tag
    #print (viterbi_chart)
    for i in range(N-3,1,-1):
        out_tag_seq.insert(0,best_tag)
        score,prev_tag=viterbi_chart[i][best_tag]
        #print (score,prev_tag)
        best_tag=prev_tag

    return out_tag_seq,perc_calls

def get_sentence_sequences(file,lang):
    """
    iterate each line 
    :return: 
    """
    line=file.readline()
    i=0
    sequence_arr = []
    sentence_arr=[]
    while line:
        line=line.strip()
        if line:
            features,goldtag=line.rsplit(' ',1)
            #word,features=_.split(' ',1)
            #word_arr.append(word)
            sequence_arr.append((get_dicts(features,lang),goldtag))
        else:
            #i+=1
            #if i==2:
            sentence_arr.append(sequence_arr)
            i=0
            sequence_arr = []
        line = file.readline()
        #print (line)
    return sentence_arr

def update_weights(feats,ner_tag_out,labels):
    for i,feat in enumerate(feats):
        pred=ner_tag_out[i]
        gold=labels[i]
        #print (pred,gold)
        if pred !=gold:
            perc.nupdates+=1
            perc._update_weights(Instance(feat,''),gold,+1.0)
            perc._update_weights(Instance(feat,''),pred,-1.0)



def train (sentence_seq,iters,validate=False):
    start=['-START1-','-START2-']
    end = ['-END1-', '-END2-']
    feats=[]
    labels=[]
    words=[]
    k=len(sentence_seq)
    words=[tokens[0]['word'] for tokens in sentence_seq]
    for i,tokens in enumerate(sentence_seq):
        if i==0:
            prev={'word':'-START1-','pos_tag':'O','chunk_tag':'O'}
            prev2={'word':'-START2-','pos_tag':'O','chunk_tag':'O'}
        elif i==1:
            prev2={'word':'-START1-','pos_tag':'O','chunk_tag':'O'}
            prev=tokens[0]
        else:
            prev2=sentence_seq[i-2][0]
            prev=sentence_seq[i-1][0]

        if i==k-1:
            next={'word':'-END1-','pos_tag':'O','chunk_tag':'O'}
            next2 = {'word': '-END2-', 'pos_tag': 'O', 'chunk_tag': 'O'}
        elif i==k-2:
            next=sentence_seq[i+1][0]
            next2={'word':'-END1-','pos_tag':'O','chunk_tag':'O'}
        else:
            next = sentence_seq[i+1][0]
            next2 = sentence_seq[i+2][0]

        feats.append(get_features(i,tokens[0],prev,prev2,next,next2,words))
        labels.append(tokens[1])

        #print (feats)
    #print (feats)
    ner_tag_out,t_perc_calls=viterbi(feats,words)
    #print ("NER - ",ner_tag_out)
    if validate==False:
        update_weights(feats,ner_tag_out,labels)
        return t_perc_calls
    else :
        return (ner_tag_out,feats,labels)


def get_tagdict(sentences):
    counts=defaultdict(lambda: defaultdict(int))
    tagdict={}
    wordcounts=Counter()
    for sent_seq in sentences:
        for i,wordcont in enumerate(sent_seq):
            word=sent_seq[i][0]['word']
            tag=sent_seq[i][1]
            counts[word][tag]+=1
            wordcounts.update([word])
    min_count=20
    acceptable_perc=0.9
    for word,tag_count in counts.items():
        tag,maxcount=max(tag_count.items(),key=lambda x: x[1])
        n=sum(tag_count.values())
        if n>=min_count and float(maxcount/n) >= acceptable_perc:
            tagdict[word]=tag
    return tagdict,wordcounts

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

if __name__=='__main__':
    tag_set = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-MISC', 'I-MISC', 'O']
    sentences=[]
    feat_array=[]
    iters=10
    perc=Perceptron([Instance([],'')])
    perc.nupdates=0
    total_instances=0
    tagdict={}
    wordcounts={}
    lang='deu'
    loc_gaz = set()
    with open(r"GB/GB.txt", encoding='utf-8') as f:
        line = f.readline()
        while line:
            loc = line.strip().split("\t")[1]
            loc_gaz.update([loc])
            line = f.readline()
    if lang=='eng':
       input_train = r"./conll03/eng.train"
       testb=r"./conll03/eng.testb"
       testa = r"./conll03/eng.testa"
       test_out=r"./conll03/eng.testb_out"
    elif lang=='deu':
        input_train = r"./conll03/deu.train"
        testb = r"./conll03/deu.testb"
        testa = r"./conll03/deu.testa"
        test_out = r"./conll03/deu.testb_out"

    for x in range(iters):
        with open(input_train) as f:
            sentences=get_sentence_sequences(f,lang)
            tagdict,wordcounts=get_tagdict(sentences)
        random.shuffle(sentences)
        i=0
        for sent in sentences:
           #print (seq)
           #print (seq)
            perc_calls=train(sent,1)
            total_instances += perc_calls
            i+=1
            if i%3000==0:
                print ("iteration:",x," record: ",i)
        print ("end of iteration ",x," total updates ",perc.nupdates,total_instances)
    perc._average(total_instances)

    pred_seq=[]
    actual_seq=[]
    with open (testb) as f:
        test_sentences=get_sentence_sequences(f,lang)
    for test_sent in test_sentences:
        pred,feats,actual=train(test_sent,1,validate=True)
        pred_seq+=pred+[' ']
        actual_seq+=actual+[' ']
        #print (feats , actual,pred)
        k=0
    with open (test_out,"w") as f:
        with open(testb) as f2:
            x=f2.readline()
            while x:
                x=x.strip()
                x=x+' '+pred_seq[k]+'\n'
                f.write(x)
                k+=1
                x = f2.readline()

    print (np.unique(actual_seq,return_counts=True))
    cm=confusion_matrix(actual_seq,pred_seq,labels=tag_set)
    print_cm(cm,tag_set)










