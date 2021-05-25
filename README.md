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

You can directly download our preprocessed train & dev data of HotpotQA from the [link](https://drive.google.com/drive/folders/1zV03LosHm55RLOJoRoF0KwXB9bm7Z7zH?usp=sharing) 

Extract all compressed files into **./hotpot/** folder.

Also you can preprocess by yourself following the instructions in the next section.

### Preprocess 

Previously we provided intermediate data files for training MulQG. Now you can also run the following preprocessing.
The preprocessing phase consists of paragraph selection, named entity recognition, and graph construction.


* Step 1.1: First, download model checkpoints and save them in **./work_dir** 
  - [bert_ner.pt](https://drive.google.com/file/d/1KneaQDpZ3uWXUEQCS-nC4OQQT8VBsLJr/view?usp=sharing)
  - [para_select_model.bin](https://drive.google.com/file/d/10kTPjd-OXXROzqAyz8vVuWBNvuwolHA1/view?usp=sharing)


* Step 2: Run the data preprocessing (change the input and output path to your own)
```
sh ./run_preprocess.sh
```

* Step 3: Run the process_hotpot.py (to obtain the `embedding.pkl` and `word2idx.pkl`)


### Released Checkpoints

We also released our pretrained model for reproduction.
* [MulQG_BFS.tar.gz](https://drive.google.com/file/d/1NCMDg8j3VsvQ3ul1FjBk6l_TT9c7urQB/view?usp=sharing)

### Training

* Step 4: Run the training  

```
sh ./run_train.sh 
```

```
python3 -m GPG.main --use_cuda --schedule --ans_update --q_attn --is_coverage --use_copy --batch_size=36 --beam_search --gpus=0,1,2 --position_embeddings_flag

```
Change the configuration file in *GPG/config.py* with proper data path, eg, the log path, the output model path, so on.
If an OOM exception occurs, you may try to set a smaller batch size with gradient_accumulate_step > 1.
Your checkpoints in each epoch will be stored in  *./output/* directory respectively. or you can change the path in *GPG/config.py*.

### Inference

* Step 5: Do the inference, and the prediction results will be under *./prediction/*  (you may modify other configurations in *GPG/config.py* file)
```
sh ./run_inference.sh
```
You can do the inference using our released model [MulQG_BFS.tar.gz](https://drive.google.com/file/d/1NCMDg8j3VsvQ3ul1FjBk6l_TT9c7urQB/view?usp=sharing), with the following command:

```
python3 -m GPG.main --notrain --max_tgt_len=30 --min_tgt_len=8 --beam_size=10 --use_cuda --ans_update --q_attn --is_coverage --use_copy --batch_size=20 --beam_search --gpus=0 --position_embeddings_flag --restore="./output/GraphPointerGenerator/MulQG_BFS_checkpoint.pt"

```

### Evaluation

* Step 6: Do the evaluation. We calculate the **BLEU** and **METOR**, and **ROUGE** score via [**nlg-eval**](https://github.com/Maluuba/nlg-eval), and the **Answerability** and **QBLEU** metrics via [Answeriblity-Metric](https://github.com/PrekshaNema25/Answerability-Metric). You may need to install them.

We also upload our prediction output as in *./prediction/* directory, using [**nlg-eval**](https://github.com/Maluuba/nlg-eval) packages via:

```
nlg-eval --hypothesis=./prediction/candidate.txt --references=./predictioin/golden.txt
```

you will get *nlg-eval* results like:

```
Bleu_1: 0.401475
Bleu_2: 0.267142
Bleu_3: 0.197256
Bleu_4: 0.151990
METEOR: 0.205085
ROUGE_L: 0.352992
```

Also, follow the instructions [Answeriblity-Metric](https://github.com/PrekshaNema25/Answerability-Metric) to measure the *Answerability* and *QBLEU* metircs.

```
python3 answerability_score.py --data_type squad --ref_file ./prediction/golden.txt --hyp_file ./prediction/candidate.txt --ner_weight 0.6 --qt_weight 0.2 --re_weight 0.1 --delta 0.7 --ngram_metric Bleu_4
```
then you will get the *QBLEU4* as in (according to the paper, this should be the QBLEU-4 value, just ignore the words)

## Installation

This repo is tested on Python 3.6+, PyTorch 1.0.0+ (PyTorch 1.3.1+ for examples) and TensorFlow 2.0.

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Create a virtual environment with the version of Python you're going to use and activate it.

Now, if you want to use 🤗 Transformers, you can install it with pip. If you'd like to play with the examples, you must install it from source.

### With pip

First you need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```

### From source

Here also, you first need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/get-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

When you update the repository, you should upgrade the transformers installation and its dependencies as follows:

```bash
git pull
pip install --upgrade .
```

### Run the examples

Examples are included in the repository but are not shipped with the library.

Therefore, in order to run the latest versions of the examples, you need to install from source, as described above.

Look at the [README](https://github.com/huggingface/transformers/blob/master/examples/README.md) for how to run examples.

### Tests

A series of tests are included for the library and for some example scripts. Library tests can be found in the [tests folder](https://github.com/huggingface/transformers/tree/master/tests) and examples tests in the [examples folder](https://github.com/huggingface/transformers/tree/master/examples).

Depending on which framework is installed (TensorFlow 2.0 and/or PyTorch), the irrelevant tests will be skipped. Ensure that both frameworks are installed if you want to execute all tests.

Here's the easiest way to run tests for the library:

```bash
pip install -e ".[testing]"
make test
```

and for the examples:

```bash
pip install -e ".[testing]"
pip install -r examples/requirements.txt
make test-examples
```

For details, refer to the [contributing guide](https://github.com/huggingface/transformers/blob/master/CONTRIBUTING.md#tests).

### Do you want to run a Transformer model on a mobile device?

You should check out our [`swift-coreml-transformers`](https://github.com/huggingface/swift-coreml-transformers) repo.

It contains a set of tools to convert PyTorch or TensorFlow 2.0 trained Transformer models (currently contains `GPT-2`, `DistilGPT-2`, `BERT`, and `DistilBERT`) to CoreML models that run on iOS devices.

At some point in the future, you'll be able to seamlessly move from pre-training or fine-tuning models to productizing them in CoreML, or prototype a model or an app in CoreML then research its hyperparameters or architecture from TensorFlow 2.0 and/or PyTorch. Super exciting!

## Model architectures

🤗 Transformers currently provides the following NLU/NLG architectures:

1. **[BERT](https://huggingface.co/transformers/model_doc/bert.html)** (from Google) released with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.
2. **[GPT](https://huggingface.co/transformers/model_doc/gpt.html)** (from OpenAI) released with the paper [Improving Language Understanding by Generative Pre-Training](https://blog.openai.com/language-unsupervised/) by Alec Radford, Karthik Narasimhan, Tim Salimans and Ilya Sutskever.
3. **[GPT-2](https://huggingface.co/transformers/model_doc/gpt2.html)** (from OpenAI) released with the paper [Language Models are Unsupervised Multitask Learners](https://blog.openai.com/better-language-models/) by Alec Radford*, Jeffrey Wu*, Rewon Child, David Luan, Dario Amodei** and Ilya Sutskever**.
4. **[Transformer-XL](https://huggingface.co/transformers/model_doc/transformerxl.html)** (from Google/CMU) released with the paper [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) by Zihang Dai*, Zhilin Yang*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov.
5. **[XLNet](https://huggingface.co/transformers/model_doc/xlnet.html)** (from Google/CMU) released with the paper [​XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) by Zhilin Yang*, Zihang Dai*, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le.
6. **[XLM](https://huggingface.co/transformers/model_doc/xlm.html)** (from Facebook) released together with the paper [Cross-lingual Language Model Pretraining](https://arxiv.org/abs/1901.07291) by Guillaume Lample and Alexis Conneau.
7. **[RoBERTa](https://huggingface.co/transformers/model_doc/roberta.html)** (from Facebook), released together with the paper a [Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) by Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.
8. **[DistilBERT](https://huggingface.co/transformers/model_doc/distilbert.html)** (from HuggingFace), released together with the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) by Victor Sanh, Lysandre Debut and Thomas Wolf. The same method has been applied to compress GPT2 into [DistilGPT2](https://github.com/huggingface/transformers/tree/master/examples/distillation), RoBERTa into [DistilRoBERTa](https://github.com/huggingface/transformers/tree/master/examples/distillation), Multilingual BERT into [DistilmBERT](https://github.com/huggingface/transformers/tree/master/examples/distillation) and a German version of DistilBERT.
9. **[CTRL](https://huggingface.co/transformers/model_doc/ctrl.html)** (from Salesforce) released with the paper [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and Richard Socher.
10. **[CamemBERT](https://huggingface.co/transformers/model_doc/camembert.html)** (from Inria/Facebook/Sorbonne) released with the paper [CamemBERT: a Tasty French Language Model](https://arxiv.org/abs/1911.03894) by Louis Martin*, Benjamin Muller*, Pedro Javier Ortiz Suárez*, Yoann Dupont, Laurent Romary, Éric Villemonte de la Clergerie, Djamé Seddah and Benoît Sagot.
11. **[ALBERT](https://huggingface.co/transformers/model_doc/albert.html)** (from Google Research and the Toyota Technological Institute at Chicago) released with the paper [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942), by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut.
12. **[T5](https://huggingface.co/transformers/model_doc/t5.html)** (from Google AI) released with the paper [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.
13. **[XLM-RoBERTa](https://huggingface.co/transformers/model_doc/xlmroberta.html)** (from Facebook AI), released together with the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) by Alexis Conneau*, Kartikay Khandelwal*, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer and Veselin Stoyanov.
14. **[MMBT](https://github.com/facebookresearch/mmbt/)** (from Facebook), released together with the paper a [Supervised Multimodal Bitransformers for Classifying Images and Text](https://arxiv.org/pdf/1909.02950.pdf) by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.
15. **[FlauBERT](https://huggingface.co/transformers/model_doc/flaubert.html)** (from CNRS) released with the paper [FlauBERT: Unsupervised Language Model Pre-training for French](https://arxiv.org/abs/1912.05372) by Hang Le, Loïc Vial, Jibril Frej, Vincent Segonne, Maximin Coavoux, Benjamin Lecouteux, Alexandre Allauzen, Benoît Crabbé, Laurent Besacier, Didier Schwab.
16. **[BART](https://huggingface.co/transformers/model_doc/bart.html)** (from Facebook) released with the paper [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf) by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer.
17. **[ELECTRA](https://huggingface.co/transformers/model_doc/electra.html)** (from Google Research/Stanford University) released with the paper [ELECTRA: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/abs/2003.10555) by Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning.
18. **[DialoGPT](https://huggingface.co/transformers/model_doc/dialogpt.html)** (from Microsoft Research) released with the paper [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) by Yizhe Zhang, Siqi Sun, Michel Galley, Yen-Chun Chen, Chris Brockett, Xiang Gao, Jianfeng Gao, Jingjing Liu, Bill Dolan.
19. **[Reformer](https://huggingface.co/transformers/model_doc/reformer.html)** (from Google Research) released with the paper [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) by Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya.
20. **[MarianMT](https://huggingface.co/transformers/model_doc/marian.html)** Machine translation models trained using [OPUS](http://opus.nlpl.eu/) data by Jörg Tiedemann. The [Marian Framework](https://marian-nmt.github.io/) is being developed by the Microsoft Translator Team.
21. **[Longformer](https://huggingface.co/transformers/model_doc/longformer.html)** (from AllenAI) released with the paper [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) by Iz Beltagy, Matthew E. Peters, Arman Cohan.
22. **[DPR](https://github.com/facebookresearch/DPR)** (from Facebook) released with the paper [Dense Passage Retrieval
for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) by Vladimir Karpukhin, Barlas Oğuz, Sewon
Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih.
23. **[Pegasus](https://github.com/google-research/pegasus)** (from Google) released with the paper [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)> by Jingqing Zhang, Yao Zhao, Mohammad Saleh and Peter J. Liu.
24. **[MBart](https://github.com/pytorch/fairseq/tree/master/examples/mbart)** (from Facebook) released with the paper  [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) by Yinhan Liu, Jiatao Gu, Naman Goyal, Xian Li, Sergey Edunov, Marjan Ghazvininejad, Mike Lewis, Luke Zettlemoyer.  
25. **[Other community models](https://huggingface.co/models)**, contributed by the [community](https://huggingface.co/users).
26. Want to contribute a new model? We have added a **detailed guide and templates** to guide you in the process of adding a new model. You can find them in the [`templates`](./templates) folder of the repository. Be sure to check the [contributing guidelines](./CONTRIBUTING.md) and contact the maintainers or open an issue to collect feedbacks before starting your PR.

These implementations have been tested on several datasets (see the example scripts) and should match the performances of the original implementations (e.g. ~93 F1 on SQuAD for BERT Whole-Word-Masking, ~88 F1 on RocStories for OpenAI GPT, ~18.3 perplexity on WikiText 103 for Transformer-XL, ~0.916 Pearson R coefficient on STS-B for XLNet). You can find more details on the performances in the Examples section of the [documentation](https://huggingface.co/transformers/examples.html).


## Citation

We now have a [paper](https://arxiv.org/abs/1910.03771) you can cite for the 🤗 Transformers library:
```bibtex
@article{Wolf2019HuggingFacesTS,
  title={HuggingFace's Transformers: State-of-the-art Natural Language Processing},
  author={Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond and Clement Delangue and Anthony Moi and Pierric Cistac and Tim Rault and Rémi Louf and Morgan Funtowicz and Joe Davison and Sam Shleifer and Patrick von Platen and Clara Ma and Yacine Jernite and Julien Plu and Canwen Xu and Teven Le Scao and Sylvain Gugger and Mariama Drame and Quentin Lhoest and Alexander M. Rush},
  journal={ArXiv},
  year={2019},
  volume={abs/1910.03771}
}
```
