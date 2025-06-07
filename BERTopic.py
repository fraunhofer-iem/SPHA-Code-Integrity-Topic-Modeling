import configparser
import json
import os
import sys
import logging
import joblib

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# --- Utility Functions ---
def load_config(config_path: str) -> configparser.ConfigParser:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def load_json_messages(json_path: str) -> list[str]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not (isinstance(data, list) and all(isinstance(entry, dict) and 'Message' in entry for entry in data)):
        raise ValueError(f"JSON data at '{json_path}' is not in the expected format.")

    return [entry['Message'].replace("-", " ").lower() for entry in data]


def load_embeddings(embedding_path: str) -> np.ndarray:
    return np.load(embedding_path)


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def get_umap_hdbscan_models(min_cluster_size: int):
    try:
        from cuml.manifold import UMAP as cumlUMAP
        from cuml.cluster import HDBSCAN as cumlHDBSCAN
        logging.info("cuML found. Using GPU for UMAP and HDBSCAN.")
        return (
            cumlUMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42),
            cumlHDBSCAN(min_cluster_size=min_cluster_size, min_samples=1, metric='euclidean',
                        gen_min_span_tree=True, prediction_data=True)
        )
    except:
        from umap import UMAP
        from hdbscan import HDBSCAN
        logging.info("cuML not found. Using CPU for UMAP and HDBSCAN.")
        return (
            UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42),
            HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean',
                    cluster_selection_method='eom', prediction_data=True)
        )


def save_results(output_dir: str, topic_model: BERTopic, model_name: str, commit_messages: list[str], topics, topic_info: pd.DataFrame):
    os.makedirs(output_dir, exist_ok=True)
    topic_model.save(os.path.join(output_dir, "bertopic_model"), serialization="safetensors",save_ctfidf=True, save_embedding_model=model_name)

    pd.DataFrame({'Message': commit_messages, 'Topic': topics}).to_csv(os.path.join(output_dir, "commit_topic_assignments.csv"), index=False)

    if topic_info is not None:
        topic_info.to_csv(os.path.join(output_dir, "topic_information.csv"), index=False)


# --- Main ---
def main():
    # The first argument is always the script name
    if len(sys.argv) > 1:
        print("First argument:", sys.argv[1])
    else:
        print("No arguments provided.")
        exit(1)
    try:
        config = load_config(sys.argv[1])

        # Paths
        output_dir = config.get('Paths', 'results_output_dir')
        os.makedirs(output_dir, exist_ok=True)

        umap_dir = os.path.join(output_dir, config.get('Paths', 'umap_dir'))
        hdbscan_dir = os.path.join(output_dir, config.get('Paths', 'hdbscan_dir'))

        umap_model_path = os.path.join(umap_dir, "umap_fitted.joblib")
        hdbscan_model_path = os.path.join(hdbscan_dir, "hdbscan_fitted.joblib")

        logging.info("Loading embeddings...")
        json_path = config.get('Paths', 'json_file_path')
        embedding_path = config.get('Paths', 'embeddings_path')
        model_name = config.get('Models', 'embedding_model_name')

        ngram_lower = config.getint('BERTopicParams', 'ngram_range_lower')
        ngram_upper = config.getint('BERTopicParams', 'ngram_range_upper')
        mmr_diversity = config.getfloat('BERTopicParams', 'mmr_diversity')
        min_cluster_size = config.getint('BERTopicParams', 'hdbscan_min_cluster_size')

        logging.info("Configuration loaded.")

        commit_messages = load_json_messages(json_path)
        embeddings = load_embeddings(embedding_path)

        umap = joblib.load(umap_model_path)
        hdbscan = joblib.load(hdbscan_model_path)

        logging.info(f"{len(commit_messages)} commit messages and embeddings loaded.")

        embedding_model = load_embedding_model(model_name)
        umap_model, hdbscan_model = get_umap_hdbscan_models(min_cluster_size=min_cluster_size)

        representation_model = [
            MaximalMarginalRelevance(diversity=mmr_diversity),
            KeyBERTInspired()
        ]

        vectorizer = CountVectorizer(stop_words="english", ngram_range=(ngram_lower, ngram_upper))
        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

        topic_model = BERTopic(
            vectorizer_model=vectorizer,
            umap_model=umap,
            hdbscan_model=hdbscan,
            embedding_model=embedding_model,
            representation_model=representation_model,
            ctfidf_model=ctfidf,
            nr_topics="auto",
            verbose=True
        )

        topics, _ = topic_model.fit_transform(commit_messages, embeddings)
        topic_info = topic_model.get_topic_info()

        logging.info(f"Discovered {len(topic_model.get_topics())} topics (including outliers).")
        logging.info(f"Outliers: {list(topics).count(-1)}")

        save_results(output_dir, topic_model, model_name, commit_messages, topics, topic_info)
        fig = topic_model.visualize_topics()
        fig.write_html(os.path.join(output_dir, "result.html"))

        logging.info("All results saved.")

    except Exception as e:
        logging.error(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
