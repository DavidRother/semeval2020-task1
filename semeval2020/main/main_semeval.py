from semeval2020.factory_hub import model_factory
from semeval2020.factory_hub import preprocessor_factory, config_factory
from semeval2020.util import preprocessing

import os.path
import shutil
import warnings
import pprint
import numpy as np
import tqdm
from numba import NumbaPerformanceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

################################################
# Pipeline architecture ########################
################################################

model_name = "HDBSCAN"
preprocessing_method = "UMAP"
embedding_type = 'bert'

################################################
# Configs ######################################
################################################

config_paths = "ProjectPaths"

################################################
# Task Specifications ##########################
################################################

tasks = ("task1", "task2")
languages = ("english", "latin", "german", "swedish")
corpora = ("corpus1", "corpus2")

################################################
#  Code ########################################
################################################

paths = config_factory.get_config(config_paths)
base_path = paths['base_path']
target_file = f"{base_path}targets/english.txt"
with open(target_file, 'r', encoding='utf-8') as f_in:
    english_target_words = [line.strip().split('\t')[0] for line in f_in]

task_params = config_factory.get_config("TaskParameter")

answer_dict = {"task1": {}, "task2": {}}
label_encoding = {corpus: idx for idx, corpus in enumerate(corpora)}

# Compute the answers

for lang_idx, language in enumerate(languages):
    target_file = f"{base_path}targets/{language}.txt"
    with open(target_file, 'r', encoding='utf-8') as f_in:
        target_words = [line.strip().split('\t')[0] for line in f_in]
        if language == "english":
            target_words = [preprocessing.remove_pos_tagging_word(word) for word in target_words]

    answer_dict["task1"][language] = {}
    answer_dict["task2"][language] = {}
    k = task_params[language]["k"]
    n = task_params[language]["n"]

    for fig_idx, word in tqdm.tqdm(enumerate(target_words)):
        file1 = f"{paths[f'auto_embedding_{embedding_type}']}{language}/corpus1/{word}.npy"
        auto_embedded_data1 = np.load(file1)
        file2 = f"{paths[f'auto_embedding_{embedding_type}']}{language}/corpus2/{word}.npy"
        auto_embedded_data2 = np.load(file2)
        embeddings_label_encoded = []
        embeddings_label_encoded.extend([0] * len(auto_embedded_data1))
        embeddings_label_encoded.extend([1] * len(auto_embedded_data2))

        x_data = np.vstack([auto_embedded_data1, auto_embedded_data2])

        preprocessor = preprocessor_factory.create_preprocessor(preprocessing_method,
                                                                **config_factory.get_config(preprocessing_method)[
                                                                    language])
        x_data = preprocessor.fit_transform(x_data)

        model = model_factory.create_model(model_name, **config_factory.get_config(model_name)[language])
        task_1_answer, task_2_answer = model.fit_predict(x_data, embedding_epochs_labeled=embeddings_label_encoded,
                                                         k=k, n=n)

        answer_dict["task1"][language][word] = task_1_answer
        answer_dict["task2"][language][word] = task_2_answer

##############################################
# Evaluate the answer and save it ############
##############################################

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(answer_dict)

for task in answer_dict:
    for language in answer_dict[task]:

        task_path = f"{paths['answer_path_main']}{task}/"
        os.makedirs(task_path, exist_ok=True)

        with open(f"{task_path}{language}.txt", 'w', encoding='utf-8') as f_out:
            for word in answer_dict[task][language]:
                answer = int(answer_dict[task][language][word]) if task == "task1" else \
                    float(answer_dict[task][language][word])
                if language == "english":
                    for t_word in english_target_words:
                        if word == t_word[:-3]:
                            word = t_word
                f_out.write('\t'.join((word, str(answer) + '\n')))

shutil.make_archive(paths['out_zip_path_main'], 'zip', paths['in_zip_path_main'])

print("done")
