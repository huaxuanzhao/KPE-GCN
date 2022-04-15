import os
import time
import random
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import *
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

start_time = time.time()

dataset = 'proposal'

# read label file --> doc_name_list
# read content file -->doc_content_list
doc_name_list = []
f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
f.close()

doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

# imbalance dataset --> label_dict
label_dict = {}
for i,doc_meta in enumerate(doc_name_list):
    label = doc_meta.split('\t')[2].strip()
    if label in label_dict:
        label_list_temp = label_dict[label]
        label_list_temp.append(i)
        label_dict[label] = label_list_temp
    else:
        label_dict[label] = [i]

num_keywordSet = 600
for key,item in label_dict.items():
    print("label：{}, num ：{}".format(key, len(item)))
    str_temp = ''
    for i in item:
        str_temp = str_temp + doc_content_list[i]
    keyword_dict_temp = extract_keywords(str_temp)
    keyword_dict_temp = sorted(keyword_dict_temp.items(), key=lambda kv: (kv[1], kv[0]),reverse=True)
    print(len(keyword_dict_temp))
    if len(keyword_dict_temp) > num_keywordSet:
        cut_num = num_keywordSet
    else:
        cut_num = len(keyword_dict_temp)
    keyword_dict = listcut(keyword_dict_temp, 0, cut_num)
    with open('data/keywords/label_' + str(key) + '_keyword_dict_tttt.txt', 'w') as file:
        file.write(json.dumps(keyword_dict))

end_time = time.time()
print(f"Runing TIME: {end_time - start_time} s\n")