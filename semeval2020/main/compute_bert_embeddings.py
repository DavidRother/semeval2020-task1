from semeval2020.language_models.bertwrapper import BertWrapper
from semeval2020.data_loader.sentence_loader import SentenceLoader
from semeval2020.util import preprocessing
from semeval2020.factory_hub import config_factory

import numpy as np
import torch
import os.path
import tqdm


########################################
#  Config Parameter ####################
########################################

paths = config_factory.get_config("ProjectPaths")

language = 'english'
corpus = "corpus2"

base_path = paths["base_path"]
model_string = "bert-base-multilingual-cased"

output_path = paths["bert_embeddings"]

max_length_sentence = 70
padding_length = 100

########################################
#  Code ################################
########################################

base_out_path = f"{output_path}{language}/{corpus}/"
os.makedirs(base_out_path, exist_ok=True)

data_loader = SentenceLoader(base_path, language=language, corpus=corpus)
bert_model = BertWrapper(model_string=model_string)

# load the data and the target words
target_words, sentences = data_loader.load()

# prepare the sentences
sentences = preprocessing.sanitized_sentences(sentences, max_len=max_length_sentence)
sentences = preprocessing.filter_for_words(sentences, target_words)
sentences = preprocessing.remove_pos_tagging(sentences, target_words)
sentences = preprocessing.remove_numbers(sentences)

target_words = [preprocessing.remove_pos_tagging_word(word) for word in target_words]

# tokenize sentences
tokenized_target_sentences = bert_model.tokenize_sentences(sentences, target_words)

# allocate dicts to save embeddings and sentences mapped to target words
target_embeddings_dict = {target: [] for target in target_words}
target_sentences_dict = {target: [] for target in target_words}

bert_model.enter_eval_mode()

# compute the embeddings
for tokenized_sentence, target_word_idx_dict in tqdm.tqdm(tokenized_target_sentences):
    input_ids = bert_model.get_tokenized_input_ids(tokenized_sentence, padding_length=padding_length)
    attention_mask = bert_model.get_attention_mask(input_ids)

    input_id_tensor = torch.tensor([input_ids])
    attention_mask_tensor = torch.tensor([attention_mask])
    target_embeddings = bert_model.compute_embeddings(input_id_tensor, attention_mask_tensor, target_word_idx_dict)
    for target, embeddings in target_embeddings.items():
        target_embeddings_dict[target].extend(embeddings)
        sent = ' '.join(tokenized_sentence)
        target_sentences_dict[target].extend([sent] * len(embeddings))


# save the embeddings and the sentences
os.makedirs(base_out_path, exist_ok=True)

for target, target_embeddings in target_embeddings_dict.items():
    np.save(f"{base_out_path}{target}", target_embeddings)

