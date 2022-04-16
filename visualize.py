from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

dataset = 'proposal'
models = ['textgcn', 'kwgcn', 'balanced_kwgcn'] 
model = models[2]

dimension = 2 # 2 or 3
f = open('./data/'+ dataset + model + '_shuffle.txt', 'r')
lines = f.readlines()
f.close()

f = open('./data/'+ dataset + model + '_doc_vectors.txt', 'r')
embedding_lines = f.readlines()
f.close()

target_names = set()
labels = []
docs = []
true_doc = []

for i in range(len(lines)):
    line = lines[i].strip()
    temp = line.split('\t')
    if temp[0].startswith('R'):
        true_doc.append(0)
    else:
        true_doc.append(1)
    labels.append(int(temp[2]))
    emb_str = embedding_lines[i].strip().split()
    values_str_list = emb_str[1:]
    values = [float(x) for x in values_str_list]
    docs.append(values)
    target_names.add(temp[2])

target_names = list(target_names)
true_docs = np.array(true_doc)
label = np.array(labels)

# TSNE
fea = TSNE(n_components=dimension).fit_transform(docs)
pdf = PdfPages('result/'+ dimension + 'd_' + model + '_doc_visualize.pdf') # 3d_balanced_kwgcn_doc_visualize.pdf
cls = np.unique(label)

fea_num = []
for i in cls:
    fea_num_temp = []
    for j in range(len(label)):
        if label[j]==i and true_docs[j]==1:
            fea_num_temp.append(fea[j])
    fea_num_temp = np.array(fea_num_temp)
    fea_num.append(fea_num_temp)
fea_num = np.array(fea_num)

fig = plt.figure()
if dimension == 3:
    ax = Axes3D(fig)
for i, f in enumerate(fea_num):
    if cls[i] in range(10): 
        if dimension == 2:
            plt.scatter(f[:, 0], f[:, 1], alpha = 0.5, s=12)
        elif dimension == 3:
            ax.scatter(f[:, 0], f[:, 1], f[:, 2], alpha = 0.5, s=12)

if dimension == 2：
    plt.tight_layout()
elif dimension == 3:
    ax.grid(True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
pdf.savefig()
plt.show()
pdf.close()
