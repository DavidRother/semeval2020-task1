from gensim.models.word2vec import PathLineSentences


class Dataloader:

    def __init__(self, base_path, language='german', corpus='corpus2'):
        self.language = language
        self.corpus = corpus
        self.base_path = base_path
        self.target_words = self.parse_target_words()
        self.sentences = self.parse_corpus()

    def parse_target_words(self):
        target_file = f"{self.base_path}targets/{self.language}.txt"
        with open(target_file, 'r', encoding='utf-8') as f_in:
            return [line.strip().split('\t')[0] for line in f_in]

    def parse_corpus(self):
        corpus_dir = f"{self.base_path}corpora/{self.language}/{self.corpus}"
        return PathLineSentences(corpus_dir)
