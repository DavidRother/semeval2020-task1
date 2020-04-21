from semeval2020.factory_hub import config_factory


paths = {"bert_embeddings": "../../data/bert_embeddings/",
         "xlmr_embeddings": "../../data/xlmr_embeddings/",
         "auto_embedding_bert": "../../data/auto_embedded_bert/",
         "auto_embedding_xlmr": "../../data/auto_embedded_xlmr/",
         "answer_path_main": "../../my_results_main/answer/answer/",
         "out_zip_path_main": "../../my_results/answer",
         "in_zip_path_main": "../../my_results/answer/",
         "base_path": "../../data/main_task_data/"}

umap_ae_language = {"latin": {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 10},
                    "german": {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 10},
                    "english": {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 10},
                    "swedish": {"n_neighbors": 5, "min_dist": 0.0, "metric": 'cosine', "n_components": 10}}

auto_encoder = {"learning_rate": 1e-3, "weight_decay": 1e-5, "num_epochs": 200, "batch_size": 128, "input_size": 1024}

t_sne_ae_language = {"latin": {"n_components": 10, "perplexity": 50, "metric": "cosine"},
                     "german": {"n_components": 10, "perplexity": 50, "metric": "cosine"},
                     "english": {"n_components": 10, "perplexity": 50, "metric": "cosine"},
                     "swedish": {"n_components": 10, "perplexity": 50, "metric": "cosine"}}

pca_ae_language = {"latin": {"n_components": 10},
                   "german": {"n_components": 10},
                   "english": {"n_components": 10},
                   "swedish": {"n_components": 10}}

hdbscan_language = {"latin": {"min_ratio": 0.04, "max_min_cluster_size_and_samples": 80, "noise_filter": False},
                    "german": {"min_ratio": 0.04, "max_min_cluster_size_and_samples": 80, "noise_filter": False},
                    "english": {"min_ratio": 0.04, "max_min_cluster_size_and_samples": 80, "noise_filter": False},
                    "swedish": {"min_ratio": 0.04, "max_min_cluster_size_and_samples": 80, "noise_filter": False}}

dbscan_language = {"latin": {"eps": 2.5, "min_samples": 5},
                   "german": {"eps": 2.5, "min_samples": 5},
                   "english": {"eps": 2.5, "min_samples": 5},
                   "swedish": {"eps": 2.5, "min_samples": 5}}

gmm_language = {"latin": {"n_components": 3, "covariance_type": "diag", "reg_covar": 1e-3},
                "german": {"n_components": 3, "covariance_type": "diag", "reg_covar": 1e-3},
                "english": {"n_components": 3, "covariance_type": "diag", "reg_covar": 1e-3},
                "swedish": {"n_components": 3, "covariance_type": "diag", "reg_covar": 1e-3}}

task_params = {"latin": {"k": 2, "n": 5},
               "german": {"k": 2, "n": 5},
               "english": {"k": 2, "n": 5},
               "swedish": {"k": 2, "n": 5}}


config_factory.register("ProjectPaths", paths)
config_factory.register("UMAP_AE_Language", umap_ae_language)
config_factory.register("AutoEncoder", auto_encoder)
config_factory.register("TaskParameter", task_params)
config_factory.register("HDBSCANLanguage", hdbscan_language)
config_factory.register("DBSCANLanguage", dbscan_language)
config_factory.register("PCA_AE_Language", pca_ae_language)
config_factory.register("GMMLanguage", gmm_language)
