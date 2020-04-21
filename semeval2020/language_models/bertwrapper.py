from pytorch_pretrained_bert import BertTokenizer, BertModel
from keras.preprocessing.sequence import pad_sequences
import torch
import numpy as np


class BertWrapper:

    def __init__(self, model_string='bert-base-multilingual-cased'):
        self.model_string = model_string
        self.tokenizer = BertTokenizer.from_pretrained(self.model_string, do_lower_case=False)
        self.model = BertModel.from_pretrained(self.model_string)

    def enter_eval_mode(self):
        self.model.eval()

    def compute_embeddings(self, input_ids_tensor, attention_mask, target_word_idx_dict):
        target_embeddings = {target: [] for target in target_word_idx_dict}
        with torch.no_grad():
            encoded_layers, _ = self.model(input_ids_tensor, token_type_ids=None, attention_mask=attention_mask)
            embedding = encoded_layers[11]
            for target, target_idx in target_word_idx_dict.items():
                embeddings = [torch.mean(embedding[:, idx[0]:idx[1], :], dim=1).numpy().flatten() for idx in target_idx]
                target_embeddings[target].extend([emb for emb in embeddings
                                                  if not (np.isnan(np.sum(emb)) or np.sum(emb) == 0)])
        return target_embeddings

    def tokenize_sentences(self, sentences, word_to_index=None):
        word_to_index = word_to_index or []
        tokenized_target_words = {word: self.tokenizer.tokenize(word) for word in word_to_index}
        for sentence in sentences:
            tokenized_text = self.tokenizer.tokenize(' '.join(["[CLS]"] + sentence + ["[SEP]"]))
            word_to_idx_dict = {
                word: [(i, i + len(tokenized_target_words[word])) for i, tok in enumerate(tokenized_text)
                       if tokenized_text[i: i + len(tokenized_target_words[word])] ==
                       tokenized_target_words[word]] for word in word_to_index}
            yield tokenized_text, word_to_idx_dict

    def tokenize_sentences_direct_mapping(self, sentences, word_array, word_to_index=None):
        word_to_index = word_to_index or []
        sentences = list(sentences)
        tokenized_target_words = {word: self.tokenizer.tokenize(word) for word in word_to_index}
        for sentence, target_word in zip(sentences, word_array):
            tokenized_text = self.tokenizer.tokenize(' '.join(["[CLS]"] + sentence + ["[SEP]"]))
            word_to_idx_dict = {target_word: [(i, i + len(tokenized_target_words[target_word]))
                                              for i, tok in enumerate(tokenized_text)
                                              if tokenized_text[i: i + len(tokenized_target_words[target_word])] ==
                                              tokenized_target_words[target_word]]}
            yield tokenized_text, word_to_idx_dict

    def get_tokenized_input_ids(self, tokenized_text, padding_length):
        return pad_sequences([self.tokenizer.convert_tokens_to_ids(tokenized_text)], maxlen=padding_length,
                             dtype="long", truncating="post", padding="post")[0]

    @staticmethod
    def get_attention_mask(input_ids):
        return [float(i > 0) for i in input_ids]
