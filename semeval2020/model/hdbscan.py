from semeval2020.factory_hub import abstract_model, model_factory
import hdbscan
from semeval2020.model import model_utilities
from itertools import compress


class MyHDBSCAN(abstract_model.AbstractModel):

    def __init__(self, min_ratio=0.05, max_min_cluster_size_and_samples=100, noise_filter=False):
        self.hdbscan = None
        self.max_min_cluster_size_and_samples = max_min_cluster_size_and_samples
        self.min_ratio = min_ratio
        self.noise_filter = noise_filter

    def fit(self, data):
        min_cluster_size = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        min_samples = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        self.hdbscan.fit(data)

    def fit_predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        return self.predict(data, embedding_epochs_labeled, k=k, n=n)

    def predict(self, data, embedding_epochs_labeled=None, k=2, n=5):
        min_cluster_size = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        min_samples = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

        labels = self.hdbscan.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        if -1 in labels and self.noise_filter:
            indexer = [label != -1 for label in labels]
            labels = list(compress(labels, indexer))
            embedding_epochs_labeled = list(compress(embedding_epochs_labeled, indexer))
        return model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)

    def fit_predict_labeling(self, data, **kwargs):
        min_cluster_size = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        min_samples = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

        labels = self.hdbscan.fit_predict(data)
        return labels

    def predict_with_extra_return(self, data, embedding_epochs_labeled=None, k=2, n=5):
        min_cluster_size = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        min_samples = max(min(self.max_min_cluster_size_and_samples, int(self.min_ratio * len(data))), 2)
        self.hdbscan = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)

        labels = self.hdbscan.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        if -1 in labels and self.noise_filter:
            indexer = [label != -1 for label in labels]
            labels_ = list(compress(labels, indexer))
            embedding_epochs_labeled_ = list(compress(embedding_epochs_labeled, indexer))
            task_answers = model_utilities.compute_task_answers(labels_, embedding_epochs_labeled_, epoch_labels, k, n)
        else:
            task_answers = model_utilities.compute_task_answers(labels, embedding_epochs_labeled, epoch_labels, k, n)
        return task_answers[0], task_answers[1], labels

    def predict_labeling(self, data, **kwargs):
        raise NotImplementedError()


model_factory.register("HDBSCAN", MyHDBSCAN)
model_factory.register("HDBSCANLanguage", MyHDBSCAN)
