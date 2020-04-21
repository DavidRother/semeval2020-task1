from semeval2020.factory_hub import preprocessor_factory
from sklearn.manifold import TSNE

preprocessor_factory.register("TSNE", TSNE)
preprocessor_factory.register("TSNE_AE", TSNE)
preprocessor_factory.register("TSNE_AE_Language", TSNE)
