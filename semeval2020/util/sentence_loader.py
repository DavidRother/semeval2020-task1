from os import listdir
import os.path
import pandas as pd


class SentenceLoader:

    def __init__(self, base_path, language='german', corpus='corpus2', explicit_word_list=None):
        self.explicit_word_list = explicit_word_list or []
        self.language = language
        self.corpus = corpus
        self.base_path = base_path
        self.target_words = self.find_target_words()
        self.sentences = self.load_sentences()

    @staticmethod
    def _find_csv_filenames(path_to_dir, suffix=".csv"):
        filenames = listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

    def find_target_words(self):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        csv_filenames = self._find_csv_filenames(target_dir)
        return [os.path.splitext(filename)[0] for filename in csv_filenames]

    def load_sentences(self, target_word=None):
        target_dir = f"{self.base_path}{self.language}/{self.corpus}/"
        if target_word:
            sentence_dict = {target_word: None}
        else:
            sentence_dict = {target_word: None for target_word in self.target_words}
        for filename in self._find_csv_filenames(target_dir):
            if target_word:
                word = os.path.splitext(filename)[0]
                if word == target_word:
                    sentence_dict[word] = pd.read_csv(f"{target_dir}/{filename}").iloc[:, 1].values.tolist()
            else:
                word = os.path.splitext(filename)[0]
                sentence_dict[word] = pd.read_csv(f"{target_dir}/{filename}").iloc[:, 1].values.tolist()
        return sentence_dict

    def __getitem__(self, key):
        return self.sentences[key]
