from gensim.models.word2vec import PathLineSentences
from semeval2020.factory_hub import abstract_data_loader, data_loader_factory


class SentenceLoader(abstract_data_loader.AbstractDataLoader):

    def __init__(self, base_path, language='german', corpus='corpus2'):
        self.language = language
        self.corpus = corpus
        self.base_path = base_path

    def __parse_target_words(self):
        target_file = f"{self.base_path}targets/{self.language}.txt"
        with open(target_file, 'r', encoding='utf-8') as f_in:
            return [line.strip().split('\t')[0] for line in f_in]

    def __parse_corpus(self):
        corpus_dir = f"{self.base_path}corpora/{self.language}/{self.corpus}"
        return PathLineSentences(corpus_dir)

    def load(self):
        return self.__parse_target_words(), self.__parse_corpus()


data_loader_factory.register("sentence_loader", SentenceLoader)
