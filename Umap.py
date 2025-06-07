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
    return np.load(embedding_path)

def get_umap_model(n_neighbors: int, min_dist: float):
    """Return a UMAP model, attempting GPU acceleration with cuML if available."""
    try:
        from cuml.manifold import UMAP
        logging.info("cuML found. Using GPU-accelerated UMAP.")
    except ImportError:
        from umap import UMAP
        logging.info("cuML not found. Falling back to CPU-based UMAP.")

    return UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        metric='cosine',
        random_state=42
    )

def draw_umap(umap_embedding, color_data: np.ndarray, filename='umap_plot.png', title=''):
    """Draw and save a UMAP scatter plot."""
    plt.figure()
    plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=color_data, cmap='Spectral', s=5)
    plt.title(title, fontsize=18)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # Free memory


# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="UMAP projection and visualization pipeline.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    start_time = time.time()

    try:
        config = load_config(args.config_path)

        # Paths
        embedding_path = config.get('Paths', 'embeddings_path')
        output_dir = config.get('Paths', 'results_output_dir')
        os.makedirs(output_dir, exist_ok=True)

        umap_dir =  os.path.join(output_dir, config.get('Paths', 'umap_dir'))
        os.makedirs(umap_dir, exist_ok=True)
        umap_plot_path = os.path.join(umap_dir, "umap_vis.png")
        umap_model_path = os.path.join(umap_dir, "umap_fitted.joblib")

        # UMAP parameters
        n_neighbors = config.getint('BERTopicParams', 'umap_n_neighbors')
        min_dist = config.getfloat('BERTopicParams', 'umap_min_dist')

        logging.info("Loading embeddings...")
        embeddings = load_embeddings(embedding_path)

        logging.info("Initializing UMAP model...")
        umap_model = get_umap_model(n_neighbors, min_dist)

        logging.info("Fitting UMAP model...")
        umap_embedding = umap_model.fit_transform(embeddings)
        joblib.dump(umap_embedding, umap_model_path)

        logging.info("Generating UMAP visualization...")
        draw_umap(
            umap_embedding,
            color_data=embeddings[:, 0],  # Or some label/cluster assignment for better color
            filename=umap_plot_path,
            title=f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"
        )

        elapsed_time = time.time() - start_time
        logging.info("UMAP processing complete in %.2f seconds.", elapsed_time)

    except Exception:
        logging.exception("An error occurred during execution.")
        sys.exit(1)


if __name__ == '__main__':
    main()
