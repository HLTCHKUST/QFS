# Improve Query Focused Abstractive Summarization byIncorporating Answer Relevance (QFS)
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/HKUST.jpg" width="12%">

This is the implementation of the paper:

**Improve Query Focused Abstractive Summarization byIncorporating Answer Relevance**. **[Dan Su](https://github.com/Iamfinethanksu)**, Tiezheng Yu, Pascale Fung **Findings of ACL 2021** [[PDF]](https://www.aclweb.org/anthology/2020.findings-emnlp.416.pdf)

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@inproceedings{su2020multi,
  title={Multi-hop Question Generation with Graph Convolutional Network},
  author={Su, Dan and Xu, Yan and Dai, Wenliang and Ji, Ziwei and Yu, Tiezheng and Fung, Pascale},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings},
  pages={4636--4647},
  year={2020}
}
</pre>

## Abstract

Query  focused  summarization  (QFS)  modelsaim to generate summaries from source docu-ments that can answer the given query.  Mostprevious  work  on  QFS  only  considers  thequery relevance criterion when producing thesummary.  However, studying the effect of an-swer relevance in the summary generating pro-cess is also important.  In this paper, we pro-pose  QFS-BART,  a  model  that  incorporatesthe explicit answer relevance of the source doc-uments given the query via a question answer-ing  model,  to  generate  coherent  and  answer-related  summaries.   Furthermore,  our  modelcan  take  advantage  of  large  pre-trained  mod-els  which  improve  the  summarization  perfor-mance significantly.  Empirical results on theDebatepedia  dataset  show  that  the  proposedmodel achieves the new state-of-the-art perfor-mance



## Dependencies
python 3, pytorch, boto3

Or you can use `conda` environment yml file (multi-qg.yml) to create your conda environment by running
```
conda env create -f multi-qg.yml
```
or try the 
```
pip install -r requirement.txt
```

## Experiments

### Download Data

#### HotpotQA Data
Download the [hotpot QA train and test data](https://github.com/hotpotqa/hotpot) and put them under `./hotpot/data/`.

#### Glove Embedding
Download the glove embedding and unzip 'glove.840B.300d.txt' and put it under `./glove/glove.840B.300d.txt`

#### Bert Models
We use the Bert models in the paragraph selection part.
You should download and set bert pretrained model and vocabulary properly.
You can find the download links in *paragraph_selection/pytorch_pretrained_bert/modeling.py* row **40-51**, and *paragraph_selection/pytorch_pretrained_bert/tokenization.py* row **30-41**.
After you finish downloading, you should replace the dict value with your own local path accordingly.

### Preprocessed Data



### Preprocess 


### Released Checkpoints

We also released our pretrained model for reproduction.
* [MulQG_BFS.tar.gz](https://drive.google.com/file/d/1NCMDg8j3VsvQ3ul1FjBk6l_TT9c7urQB/view?usp=sharing)

### Training


### Inference



### Evaluation


## Installation

This repo is tested on Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for examples).



## Citation


