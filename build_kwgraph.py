import os
import time
import random
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

if len(sys.argv) != 2:
    sys.exit("Use: python remove_words.py <model name>")

dataset = 'proposal'
models = ['textgcn', 'kwgcn', 'balanced_kwgcn'] 
model = sys.argv[1]

if model not in models:
    sys.exit("wrong model name")

# build corpus
start_time = time.time()

word_embeddings_dim = 300
word_vector_map = {}

# shulffing
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/'+ dataset +'.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()

if model == 'balanced_kwgcn':
    f = open('data/' + dataset + '_balance.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_name_list.append(line.strip())
        temp = line.split("\t")
        if temp[1].find('test') != -1:
            doc_test_list.append(line.strip())
        elif temp[1].find('train') != -1:
            doc_train_list.append(line.strip())
    f.close()


doc_content_list = []
f = open('data/corpus/'+ dataset +'.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

len_ori = len(doc_content_list)

if model == 'balanced_kwgcn': 
    f = open('data/corpus/' + dataset + '_balance.txt', 'r')
    lines = f.readlines()
    for line in lines:
        doc_content_list.append(line.strip())
    f.close()

train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)


train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + model + '.train.index', 'w')
f.write(train_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + model + '.test.index', 'w')
f.write(test_ids_str)
f.close()

ids = train_ids + test_ids
print("proposal num: {0}".format(len(ids)))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + model + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + model + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()

# label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + model + '_labels.txt', 'w')
f.write(label_list_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
voac_doc_words_list = doc_content_list[0:len_ori]
for doc_words in voac_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    
vocab = list(word_set)
vocab_size = len(vocab)
print("vocab_size:{0}".format(vocab_size))

if model != 'textgcn':
    doc_word_str = ''.join(doc_content_list)
    keyword_dict = extract_keywords(doc_word_str)
    keyvocab = list(keyword_dict.keys())
    keyvocab_size = len(keyvocab)
    print("keyvocab_size:{0}".format(keyvocab_size))

# word doc
word_doc_list = {}
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

# count doc freq
word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)


# word ids 
word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i
vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + model + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

# keyword ids
if model != 'textgcn':
    for i in range(keyvocab_size):
        word_id_map[keyvocab[i]] = vocab_size + i 
    keyvocab_str = '\n'.join(keyvocab) 
    
    f = open('data/corpus/' + dataset + model + '_keyvocab.txt', 'w')
    f.write(keyvocab_str)
    f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set as real train, 10% training set as val

train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + model + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)


x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
# print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))
for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

if model != 'textgcn':
    keyword_vectors = np.random.uniform(-0.01, 0.01,
                                        (keyvocab_size, word_embeddings_dim))
    for i in range(len(keyvocab)):
        keyword = keyvocab[i]
        if keyword in word_vector_map:
            keyvector = word_vector_map[keyword]
            keyword_vectors[i] = keyvector
        
row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))
if model != 'textgcn':
    for i in range(keyvocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + vocab_size + train_size)) # + vocab_size 
            col_allx.append(j)
            data_allx.append(keyword_vectors.item((i, j)))



row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

if model == 'textgcn':
    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))
else:
    allx = sp.csr_matrix(
        (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size + keyvocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

if model == 'textgcn':
    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)
else:
    for i in range(vocab_size + keyvocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)
        
word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        word_i = window[i]
        if word_i in word_id_map:
            word_i_id = word_id_map[word_i]
        else:
            continue
        for j in range(0, i):
            word_j = window[j]
            if word_j in word_id_map:
                word_j_id = word_id_map[word_j]
            else:
                continue
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# word-word: pmi as weights

num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# keyword-word: ratio 
if model != 'textgcn':
    for keyword in keyvocab:
        keyword_list = keyword.split()
        length = len(keyword_list)
        keyword_id = word_id_map[keyword]
        for keyword_word in keyword_list:
            if keyword_word in vocab:
                word_id = word_id_map[keyword_word]
                row.append(train_size + keyword_id)
                col.append(train_size + word_id)
                weight.append(1.0/length)
    
# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        if word in word_id_map:
            word_id = word_id_map[word]
        else:
            continue
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

# doc-word: tf-idf
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        if word in word_id_map:
            j = word_id_map[word]
        else:
            continue
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            if model == 'textgcn':
                row.append(i + vocab_size)
            else:
                row.append(i + vocab_size + keyvocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) /
                  word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

# doc-keyword: keyscore
if model != 'textgcn':
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        for keyword,value in keyword_dict.items():
            if keyword in doc_words:
                j = word_id_map[keyword]
                if i < train_size:
                    row.append(i)
                else:
                    row.append(i + vocab_size + keyvocab_size)
                col.append(train_size + j)
                weight.append(value)

if model == 'textgcn':
    node_size = train_size + vocab_size + test_size
else:
    node_size = train_size + vocab_size + keyvocab_size + test_size

adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
f = open("data/ind.{}.x".format(dataset + model), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset + model), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset + model), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset + model), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset + model), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset + model), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset + model), 'wb')
pkl.dump(adj, f)
f.close()


end_time = time.time()
print(f"Runing TIME :{end_time - start_time} s\n")

