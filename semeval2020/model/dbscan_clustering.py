from semeval2020.factory_hub import abstract_model, model_factory
from semeval2020.model import model_utilities
from sklearn.cluster import DBSCAN
from itertools import compress


class MyDBSCAN(abstract_model.AbstractModel):

    def __init__(self, eps=1, min_samples=5):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    def fit(self, data):
        self.dbscan.fit(data)

    def fit_predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        return self.predict(data, embedding_epochs_labeled, k=k, n=n)

    def predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        labels = self.dbscan.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        if -1 in labels:
            indexer = [label != -1 for label in labels]
            labels = list(compress(labels, indexer))
            embedding_epochs_labeled = list(compress(embedding_epochs_labeled, indexer))
        return model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)

    def predict_with_extra_return(self, data, embedding_epochs_labeled=None, k=2, n=5):
        labels = self.dbscan.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        task_answers = model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)
        return task_answers[0], task_answers[1], labels

    def fit_predict_labeling(self, data, **kwargs):
        return self.dbscan.fit_predict(data, **kwargs)

    def predict_labeling(self, data, **kwargs):
        raise NotImplementedError()


model_factory.register("DBSCAN", MyDBSCAN)
model_factory.register("DBSCANLanguage", MyDBSCAN)
