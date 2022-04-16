from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from sklearn import metrics
from utils import *
from models import GCN
import random
import os
import sys
import logging


if len(sys.argv) != 2:
    sys.exit("Use: python remove_words.py <model name>")

dataset = 'proposal'
models_set = ['textgcn', 'kwgcn', 'balanced_kwgcn'] 

model_name = sys.argv[1]
if model_name not in models_set:
    sys.exit("wrong model name")
    
ckpt_dir = os.path.join("./ckpt_dir/", model_name) 
print(ckpt_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
                        
dataset_model = dataset + model_name

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, model_name +'_training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)


# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset_model , 'Dataset string.')
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset)
features = sp.identity(features.shape[0])  # featureless

print("shape of adj:{0} ".format(adj.shape))
print("shape of features:{0} ".format(features.shape))

logger.info(str(FLAGS.dataset))
# logger.info(str(FLAGS.model))

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

saver = tf.train.Saver()

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.layers[0].embedding], feed_dict=feed_dict)

    # Validation
    cost, acc, pred, labels, duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)
    logger.info("Epoch:{0}, train_loss= {1:.5f}, train_acc={2:.5f} , val_loss={3:.5f} ,val_acc={4:.5f} ,time={5:.5f}".format(epoch+1, outs[1], outs[2], cost, acc, time.time() - t))
    
    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        saver.save(sess, os.path.join(ckpt_dir, model_name + '.ckpt'))
        logger.info("Early stopping...")
        logger.info("Model saved in path: {0}".format(os.path.join(ckpt_dir, model_name + '.ckpt')))
        break

logger.info("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)

logger.info("Test set results: cost= {0:.5f}, accuracy={1:.5f} ,time={2:.5f}".format(test_cost, test_acc,test_duration))
    
test_pred = []
test_labels = []
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

logger.info("Test Precision, Recall and F1-Score...")
logger.info(str(metrics.classification_report(test_labels, test_pred, digits=4)))
logger.info("Macro average Test Precision, Recall and F1-Score...")
logger.info(str(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro')))
logger.info("Micro average Test Precision, Recall and F1-Score...")
logger.info(str(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro')))


# doc and word embeddings
logger.info('embeddings:')
word_embeddings = outs[3][train_size: adj.shape[0] - test_size] # include word and keywords
train_doc_embeddings = outs[3][:train_size]  # include train and val docs
test_doc_embeddings = outs[3][adj.shape[0] - test_size:] # test docs

logger.info(str(len(word_embeddings))+' '+str(len(train_doc_embeddings))+' '+str(len(test_doc_embeddings)))
# logger.info(str(word_embeddings))

f = open('data/corpus/' + FLAGS.dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('data/' + FLAGS.dataset + '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()

if model_name != 'textgcn':
    f = open('data/corpus/' + FLAGS.dataset + '_keyvocab.txt', 'r')
    keywords = f.readlines()
    f.close()
    
    keyvocab_size = len(keywords)
    keyword_vectors = []
    for i in range(keyvocab_size):
        keyword = keywords[i].strip()
        keyword_vector = word_embeddings[i + vocab_size]
        keyword_vector_str = ' '.join([str(x) for x in keyword_vector])
        keyword_vectors.append(keyword + ' ' + keyword_vector_str)

    keyword_embeddings_str = '\n'.join(keyword_vectors)
    f = open('data/' + FLAGS.dataset + '_keyword_vectors.txt', 'w')
    f.write(keyword_embeddings_str)
    f.close()

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
f = open('data/' + FLAGS.dataset + '_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()
