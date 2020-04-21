from semeval2020.util import sentence_loader
from semeval2020.factory_hub import config_factory

path_config = config_factory.get_config("ProjectPaths")


def map_labels_to_sentences(labels, target_word, language, corpora):
    sentences = []
    for corpus in corpora:
        loader = sentence_loader.SentenceLoader(path_config["sentence_data"], language, corpus)
        sents = loader[target_word]
        sentences.extend(sents)
    return list(zip(labels, sentences))

