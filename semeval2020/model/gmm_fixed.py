from semeval2020.factory_hub import abstract_model, model_factory
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance


class GMM(abstract_model.AbstractModel):

    def __init__(self, n_components, covariance_type, reg_covar):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.gmm = None

    def fit(self, data):
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                   reg_covar=self.reg_covar)
        self.gmm.fit(data)

    def fit_predict(self, data, embedding_epochs_labeled=None):
        self.fit(data)
        return self.predict(data, embedding_epochs_labeled)

    def predict(self, data, embedding_epochs_labeled=None):
        labels = self.gmm.predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        sense_frequencies = self.compute_cluster_sense_frequency(labels, embedding_epochs_labeled, epoch_labels)
        task_1_answer = int(any([True for sd in sense_frequencies if 0 in sense_frequencies[sd]]))
        task_2_answer = distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)
        return task_1_answer, task_2_answer

    def fit_predict_labeling(self, data, **kwargs):
        self.fit(data)
        return self.gmm.predict(data)

    def predict_with_extra_return(self, data, embedding_epochs_labeled=None, k=2, n=5):
        self.gmm = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                   reg_covar=self.reg_covar)
        labels = self.gmm.fit_predict(data)
        epoch_labels = set(embedding_epochs_labeled)
        sense_frequencies = self.compute_cluster_sense_frequency(labels, embedding_epochs_labeled, epoch_labels)
        task_1_answer = int(any([True for sd in sense_frequencies if 0 in sense_frequencies[sd]]))
        task_2_answer = distance.jensenshannon(sense_frequencies[0], sense_frequencies[1], 2.0)
        return task_1_answer, task_2_answer, labels

    def predict_labeling(self, data, **kwargs):
        return self.gmm.predict(data)

    @staticmethod
    def compute_cluster_sense_frequency(cluster_labels, embeddings_epoch_label, epoch_labels):
        n_cluster = len(set(cluster_labels))
        cluster_epoch_combined = list(zip(cluster_labels, embeddings_epoch_label))
        sense_frequencies = {epoch_label: [] for epoch_label in epoch_labels}
        for epoch in epoch_labels:
            count_epoch_total = sum(int(epoch == epoch_label) for cluster_label, epoch_label in cluster_epoch_combined)
            for sense_label in range(n_cluster):
                count_sense_epoch = sum(int(cluster_label == sense_label and epoch == epoch_label)
                                        for cluster_label, epoch_label in cluster_epoch_combined)
                sense_frequency_epoch = count_sense_epoch / count_epoch_total
                sense_frequencies[epoch].append(sense_frequency_epoch)
        return sense_frequencies


model_factory.register("GMM", GMM)
model_factory.register("GMMLanguage", GMM)

