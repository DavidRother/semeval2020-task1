# semeval2020-task1
System Implementation for Task 1 of the SemEval 2020 challenge

The website to the SemEval challenge can be found [here](https://competitions.codalab.org/competitions/20948)

Please follow the instructions on the website to get the corpus data.
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

