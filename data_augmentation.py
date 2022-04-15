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

# imbalance class
label_num_list = [0 for i in range(9)]
for key, item in label_dict.items():
    label_num_list[int(key)] = len(item)
    
# balance num list
label_balance_num = [0 for i in range(9)]
max_txt_num = max(label_num_list)
for i in range(len(label_num_list)):
    label_balance_num[i] = max_txt_num - label_num_list[i]
    
# sample keywords -->new data
new_doc_content_list = []
new_doc_name_list = []
k = 50 # sample size
for key, item in label_dict.items():
    sample_txt_num = label_balance_num[int(key)]
    if sample_txt_num==0:
        continue
    with open('data/keywords/label_' + str(key) + '_keyword_dict.txt', 'r') as file:
        keyword_dict = json.load(file)
        keywords = list(keyword_dict.keys())
        for i in range(sample_txt_num):
            # partition train/test = 6:4 same as original dataset
            if i < int(sample_txt_num * 0.6):
                train_label = 'train'
            else:
                train_label = 'test'
            single_data_sample = random.sample(keywords, k)
            single_txt = ''.join(single_data_sample)
            single_name = 'R'+str(key)+'000'+str(i)+'\t'+train_label+'\t'+str(key)
            new_doc_content_list.append(single_txt)
            new_doc_name_list.append(single_name)

new_doc_content_str = '\n'.join(new_doc_content_list)
f = open('data/' + dataset + '_balance.txt', 'w')
f.write(new_doc_content_str)
f.close()

new_doc_name_str = '\n'.join(new_doc_name_list)
f = open('data/corpus/' + dataset + '_balance.txt', 'w')
f.write(new_doc_name_str)
f.close()



end_time = time.time()
print(f"Runing TIME: {end_time - start_time} s\n")