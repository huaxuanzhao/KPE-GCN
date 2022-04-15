# KPEGCN
## Introduction
We proposes a keyphrase-enhanced graph convolution neural network (KPE-GCN) to solve the proposal classification in professional fields.
## Require
Python 3.6  
Tensorflow 1.14.0  
`RAKE`  
## Prepare work
1. Install rake following https://github.com/zelandiya/RAKE-tutorial and rename the folder `RAKE`
2. Replace the rake.py in the `RAKE` with our rake.py which add the normalization of score.
## File Structure
Please make RAKE folder and KPEGCN folder in the same directory, for example as follows.

--RAKE  
&emsp;&emsp;|----rake.py  
&emsp;&emsp;|----...  
--KPEGCN  
&emsp;&emsp;|----data # data folder  
&emsp;&emsp;&emsp;&emsp;|----corpus  # content folder  
&emsp;&emsp;&emsp;&emsp;|----keywords   
&emsp;&emsp;|----ckpt_dir # check point and log info folder    
&emsp;&emsp;&emsp;&emsp;|----training.log   
&emsp;&emsp;&emsp;&emsp;|----...      
&emsp;&emsp;|----remove_words.py  # text preprocessing   
&emsp;&emsp;|----build_kwgraph.py  # build graph for kwgcn   
&emsp;&emsp;|----train.py  # train model   
&emsp;&emsp;|----...      
   
## Reproducing Results
1. Run `python build_kwgraph.py balanced_kwgcn`
2. Run `python train.py balanced_kwgcn`
3. Change `balanced_kwgcn` in above 2 command lines to `kwgcn`, `textgcn` when producing results for other models.   
**Note**: we provide the result of `KPE-GCN`, that is `balanced_kwgcn`.  

## Example input data
1. `/data/proposal.txt` indicates proposal names, training/test split, proposal labels. Each line is for a proposal.  
2. `/data/corpus/proposal.txt` contains raw text of each proposal, each line is for the corresponding line in `/data/proposal.txt`  
3. `select_keyphrase.py` and `data_agumentation.py` belong to data agumentation part in the 'balanced_kwgcn' model. We provide a commen procedure for data agmentation which perhaps benefit for your own dataset. In our proposal dataset, '/data/proposal_balance' and '/data/corpus/proposal_balance' are the result files, which you can use directly.    
