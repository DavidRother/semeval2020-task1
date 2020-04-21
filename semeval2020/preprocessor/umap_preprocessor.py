from semeval2020.factory_hub import preprocessor_factory
import umap

preprocessor_factory.register("UMAP", umap.UMAP)
preprocessor_factory.register("UMAP_AE", umap.UMAP)
preprocessor_factory.register("UMAP_AE_Language", umap.UMAP)
