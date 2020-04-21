from semeval2020.factory_hub import preprocessor_factory
from sklearn.decomposition import PCA

preprocessor_factory.register("PCA", PCA)
preprocessor_factory.register("PCA_AE", PCA)
preprocessor_factory.register("PCA_AE_Language", PCA)
