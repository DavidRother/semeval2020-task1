# CMCE at SemEval-2020 Task 1: Clustering on Manifolds of Contextualized Embeddings to Detect Historical Meaning Shifts

This repository contains code to reproduce the experiments in our paper [CMCE at SemEval-2020 Task 1: Clustering on Manifolds of Contextualized Embeddings to Detect Historical Meaning Shifts
](https://www.aclweb.org/anthology/2020.semeval-1.22/) about measuring and detecting semantic change in the [SemEval 2020](https://competitions.codalab.org/competitions/20948) challenge. 

Please follow the instructions on the website to get the corpus data.


## CMCE
System Implementation for Task 1 of the SemEval 2020 challenge.

<!--
The website to the SemEval challenge can be found [here](https://competitions.codalab.org/competitions/20948)
-->

To use the scripts as given please create the following folder structure
with the corresponding corpora.

    semeval2020-task1/
        data/
            main_task_data/
                corpora/
                    english/
                        corpus1/
                            corpus1.txt.gz
                        corpus2/
                            corpus2.txt.gz
                    german/
                        corpus1/
                            corpus1.txt.gz
                        corpus2/
                            corpus2.txt.gz
                    latin/
                        corpus1/
                            corpus1.txt.gz
                        corpus2/
                            corpus2.txt.gz
                    swedish/
                        corpus1/
                            corpus1.txt.gz
                        corpus2/
                            corpus2.txt.gz
                targets/
                    english.txt
                    german.txt
                    latin.txt
                    swedish.txt
                

The system implemented in this repository follows 4 main steps:

1. Computing BERT or XLMR embeddings for each occurence of a word
2. Computing Auto Embeddings of the original Embeddings
3. Utilizing UMAP and HDBSCAN to perform unsupervised clustering
4. Computing the answers for the challenge given the clusters

To compute the Embeddings please execute either the compute_bert_embeddings.py or the
compute_xlmr_embeddings.py script found in the main folder and set the language and the corpus 
you need.  

Then execute the autoembed_data.py script and specify the language and the used embedding type.

At last you can compute the final task answers by executing the main_semeval.py script.
Do not forget to set the correct type of embeddings that you originally used.
(Everything defaults to BERT embeddings)

To compare the results to the truth data you can upload the created submission to the 
challenge webpage.

## Reference

```
@inproceedings{rother-etal-2020-cmce,
    title = "{CMCE} at {S}em{E}val-2020 Task 1: Clustering on Manifolds of Contextualized Embeddings to Detect Historical Meaning Shifts",
    author = "Rother, David  and
      Haider, Thomas  and
      Eger, Steffen",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.22",
    pages = "187--193",
    abstract = "This paper describes the system Clustering on Manifolds of Contextualized Embeddings (CMCE) submitted to the SemEval-2020 Task 1 on Unsupervised Lexical Semantic Change Detection. Subtask 1 asks to identify whether or not a word gained/lost a sense across two time periods. Subtask 2 is about computing a ranking of words according to the amount of change their senses underwent. Our system uses contextualized word embeddings from MBERT, whose dimensionality we reduce with an autoencoder and the UMAP algorithm, to be able to use a wider array of clustering algorithms that can automatically determine the number of clusters. We use Hierarchical Density Based Clustering (HDBSCAN) and compare it to Gaussian MixtureModels (GMMs) and other clustering algorithms. Remarkably, with only 10 dimensional MBERT embeddings (reduced from the original size of 768), our submitted model performs best on subtask 1 for English and ranks third in subtask 2 for English. In addition to describing our system, we discuss our hyperparameter configurations and examine why our system lags behind for the other languages involved in the shared task (German, Swedish, Latin). Our code is available at https://github.com/DavidRother/semeval2020-task1",
}
```
