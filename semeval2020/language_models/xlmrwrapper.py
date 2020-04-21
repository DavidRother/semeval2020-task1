import torch
import numpy as np


class XLMRWrapper:

    def __init__(self, model_string='xlmr.large'):
        self.model_string = model_string
        self.xlmr = torch.hub.load('pytorch/fairseq', self.model_string)

    def enter_eval_mode(self):
        self.xlmr.eval()

    def compute_embeddings(self, input_ids_tensor, target_word_idx_dict):
        target_embeddings = {target: [] for target in target_word_idx_dict}
        embedding = self.xlmr.extract_features(input_ids_tensor)
        for target, target_idx in target_word_idx_dict.items():
            embeddings = [torch.mean(embedding[:, idx[0]:idx[1], :], dim=1).detach().numpy().flatten() for idx in target_idx]
            target_embeddings[target].extend([emb for emb in embeddings
                                              if not (np.isnan(np.sum(emb)) or np.sum(emb) == 0)])
        return target_embeddings

    def tokenize_sentences(self, sentences, word_to_index=None):
        word_to_index = word_to_index or []
        tokenized_target_words = {word: self.xlmr.encode(word)[1:-1].tolist() for word in word_to_index}
        for sentence in sentences:
            tokenized_text = self.xlmr.encode(' '.join(sentence))
            word_to_idx_dict = {
                word: [(i, i + len(tokenized_target_words[word])) for i, tok in enumerate(tokenized_text)
                       if tokenized_text[i: i + len(tokenized_target_words[word])].tolist() ==
                       tokenized_target_words[word]] for word in word_to_index}
            yield tokenized_text, word_to_idx_dict

    def tokenize_sentences_direct_mapping(self, sentences, word_array, word_to_index=None):
        word_to_index = word_to_index or []
        sentences = list(sentences)
        tokenized_target_words = {word: self.xlmr.encode(word) for word in word_to_index}
        for sentence, target_word in zip(sentences, word_array):
            tokenized_text = self.xlmr.encode(' '.join(sentence))
            word_to_idx_dict = {target_word: [(i, i + len(tokenized_target_words[target_word]))
                                              for i, tok in enumerate(tokenized_text)
                                              if tokenized_text[i: i + len(tokenized_target_words[target_word])] ==
                                              tokenized_target_words[target_word]]}
            yield tokenized_text, word_to_idx_dict

