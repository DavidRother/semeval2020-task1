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
                

The system implemented in this repository follows 3 main steps:

1. Computing BERT or XLMR embeddings for each occurence of a word
2. Computing 


