from os import listdir
import os.path
import numpy as np
from semeval2020.factory_hub import abstract_data_loader, data_loader_factory


class LazyEmbeddingLoader(abstract_data_loader.AbstractDataLoader):

    def __init__(self, base_path, language='german', corpus='corpus2', explicit_word_list=None):
        self.explicit_word_list = explicit_word_list or []
        self.language = language
        self.corpus = corpus
        self.base_path = base_path
        self.target_words = self.find_target_words()
        self.embeddings = None

    @staticmethod
    def _find_filenames(path_to_dir, suffix=".npy"):
        filenames = listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

    def find_target_words(self):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        csv_filenames = self._find_filenames(target_dir)
        return [os.path.splitext(filename)[0] for filename in csv_filenames]

    def load_embeddings(self, target_words):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        embedding_dict = {target_word: None for target_word in target_words}
        for filename in self._find_filenames(target_dir):
            word = os.path.splitext(filename)[0]
            embedding_dict[word] = np.load(f"{target_dir}/{filename}")
        return embedding_dict

    def lazy_load_embeddings(self, word):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        for filename in self._find_filenames(target_dir):
            found_word = os.path.splitext(filename)[0]
            if found_word == word:
                return np.load(f"{target_dir}/{filename}")

    def load(self):
        target_words = self.find_target_words()
        return target_words, self.load_embeddings(target_words)

    def __getitem__(self, key):
        return self.embeddings[key]


data_loader_factory.register("lazy_embedding_loader", LazyEmbeddingLoader)
