import os
import sys
import logging
import argparse
import configparser
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --- Utility Functions ---
def load_config(config_path: str) -> configparser.ConfigParser:
    """Load and parse a configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def load_embeddings(embedding_path: str) -> np.ndarray:
    """Load embeddings from a .npy file."""
    if not os.path.isfile(embedding_path):
        raise FileNotFoundError(f"Embedding file '{embedding_path}' not found.")
    return joblib.load(embedding_path)

def get_hdbscan_model(min_cluster_size: int, min_samples: int, metric='euclidean'):
    """Return a UMAP model, attempting GPU acceleration with cuML if available."""
    try:
        from cuml.cluster import HDBSCAN
        logging.info("cuML found. Using GPU-accelerated HDBScan.")
    except ImportError:
        from hdbscan import HDBSCAN
        logging.info("cuML not found. Falling back to CPU-based HDBScan.")

    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric,
            cluster_selection_method='eom', prediction_data=True)

def draw_umap(umap_embedding, color_data: np.ndarray, filename='umap_plot.png', title=''):
    """Draw and save a UMAP scatter plot."""
    plt.figure()
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=color_data, cmap='Spectral', s=5)
    plt.title(title, fontsize=18)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="HDBScan projection and visualization pipeline.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    start_time = time.time()

    try:
        config = load_config(args.config_path)

        # Paths
        output_dir = config.get('Paths', 'results_output_dir')
        os.makedirs(output_dir, exist_ok=True)

        umap_dir = os.path.join(output_dir, config.get('Paths', 'umap_dir'))
        hdbscan_dir = os.path.join(output_dir, config.get('Paths', 'hdbscan_dir'))
        os.makedirs(hdbscan_dir, exist_ok=True)

        umap_model_path = os.path.join(umap_dir, "umap_fitted.joblib")
        hdbscan_model_path = os.path.join(hdbscan_dir, "hdbscan_fitted.joblib")

        # hdbscan parameters
        min_cluster_size = config.getint('BERTopicParams', 'hdbscan_min_cluster_size')
        min_samples = config.getint('BERTopicParams', 'hdbscan_min_samples')

        logging.info("Loading embeddings...")
        embeddings = load_embeddings(umap_model_path)

        logging.info("Initializing HDBScan model...")
        hdbscan_model = get_hdbscan_model(min_cluster_size, min_samples)

        logging.info("Fitting HDBScan model...")
        hdbscan_res = hdbscan_model.fit(embeddings)

        joblib.dump(hdbscan_res, hdbscan_model_path)

        elapsed_time = time.time() - start_time
        logging.info("HDBScan processing complete in %.2f seconds.", elapsed_time)

    except Exception:
        logging.exception("An error occurred during execution.")
        sys.exit(1)


if __name__ == '__main__':
    main()
