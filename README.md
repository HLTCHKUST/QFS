# Improve Query Focused Abstractive Summarization byIncorporating Answer Relevance (QFS)
<img src="plot/pytorch-logo-dark.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

<img align="right" src="plot/HKUST.jpg" width="12%">

This is the implementation of the paper:

**Improve Query Focused Abstractive Summarization byIncorporating Answer Relevance**. **[Dan Su](https://github.com/Iamfinethanksu)**, Tiezheng Yu, Pascale Fung **Findings of ACL 2021** [[PDF]](https://www.aclweb.org/anthology/2020.findings-emnlp.416.pdf)

If you use any source codes or datasets included in this toolkit in your work, please cite the following paper. The bibtex is listed below:
<pre>
@misc{su2021improve,
      title={Improve Query Focused Abstractive Summarization by Incorporating Answer Relevance}, 
      author={Dan Su and Tiezheng Yu and Pascale Fung},
      year={2021},
      eprint={2105.12969},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>

## Abstract

Query  focused  summarization  (QFS)  modelsaim to generate summaries from source docu-ments that can answer the given query.  Mostprevious  work  on  QFS  only  considers  thequery relevance criterion when producing thesummary.  However, studying the effect of an-swer relevance in the summary generating pro-cess is also important.  In this paper, we pro-pose  QFS-BART,  a  model  that  incorporatesthe explicit answer relevance of the source doc-uments given the query via a question answer-ing  model,  to  generate  coherent  and  answer-related  summaries.   Furthermore,  our  modelcan  take  advantage  of  large  pre-trained  mod-els  which  improve  the  summarization  perfor-mance significantly.  Empirical results on theDebatepedia  dataset  show  that  the  proposedmodel achieves the new state-of-the-art perfor-mance


## Dependencies
Our experints enviroments is:

python 3.6, pytorch(v1.6.0), transformers(v3.0.2)

Also install other dependencies via

```
pip install -r requirement.txt
```

## Experiments

### Download Data
You can download the Debatepedia data via the original [link] (https://github.com/PrekshaNema25/DiverstiyBasedAttentionMechanism)

### Released Checkpoints

We also released our pretrained model for reproduction.
* [to be released later]()

### Training/Fine-tuning QFS model

```
sh scripts/finetune_debatepedia_qfs.sh
```

### Evaluation

```
sh scripts/eval_debatepedia_qfs.sh
```


