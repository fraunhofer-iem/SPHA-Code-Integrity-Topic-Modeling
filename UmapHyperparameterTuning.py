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
    parser = argparse.ArgumentParser(description="UMAP projection and visualization pipeline for hyperparameter tuning.")
    parser.add_argument("config_path", help="Path to the configuration file.")
    args = parser.parse_args()
    total_start_time = time.time()

    try:
        config = load_config(args.config_path)

        # --- Define Parameter Grid to Iterate Over ---
        param_list = [
            {'n_neighbors': 15, 'min_dist': 0.0},
            {'n_neighbors': 15, 'min_dist': 0.001},
            {'n_neighbors': 15, 'min_dist': 0.01},
            {'n_neighbors': 15, 'min_dist': 0.05},
            {'n_neighbors': 15, 'min_dist': 0.1},
            {'n_neighbors': 100, 'min_dist': 0.0},
            {'n_neighbors': 100, 'min_dist': 0.001},
            {'n_neighbors': 100, 'min_dist': 0.01},
            {'n_neighbors': 100, 'min_dist': 0.05},
            {'n_neighbors': 100, 'min_dist': 0.1},
            {'n_neighbors': 200, 'min_dist': 0.0},
            {'n_neighbors': 200, 'min_dist': 0.001},
            {'n_neighbors': 200, 'min_dist': 0.01},
            {'n_neighbors': 200, 'min_dist': 0.05},
            {'n_neighbors': 200, 'min_dist': 0.1},
            {'n_neighbors': 400, 'min_dist': 0.0},
            {'n_neighbors': 400, 'min_dist': 0.001},
            {'n_neighbors': 400, 'min_dist': 0.01},
            {'n_neighbors': 400, 'min_dist': 0.05},
            {'n_neighbors': 400, 'min_dist': 0.1},
        ]

        # Paths from config
        embedding_path = config.get('Paths', 'embeddings_path')
        base_output_dir = config.get('Paths', 'results_output_dir')
        os.makedirs(base_output_dir, exist_ok=True)

        logging.info("Loading embeddings...")
        embeddings = load_embeddings(embedding_path)

        # --- Loop Over All Parameter Combinations ---
        for params in param_list:
            run_start_time = time.time()
            n_neighbors = params['n_neighbors']
            min_dist = params['min_dist']

            logging.info(f"--- Starting run for n_neighbors={n_neighbors}, min_dist={min_dist} ---")

            # Create a distinct directory for this run's results
            run_dir_name = f"umap_neighbors_{n_neighbors}_dist_{min_dist}"
            run_output_dir = os.path.join(base_output_dir, run_dir_name)
            os.makedirs(run_output_dir, exist_ok=True)

            umap_plot_path = os.path.join(run_output_dir, "umap_visualization.png")
            # Note: The saved object is the embedding, not the model itself, for use in BERTopic.
            # If you need the fitted model object, use joblib.dump(umap_model, ...)
            umap_embedding_path = os.path.join(run_output_dir, "umap_embedding.joblib")


            logging.info("Initializing UMAP model...")
            umap_model = get_umap_model(n_neighbors, min_dist)

            logging.info("Fitting UMAP model...")
            umap_embedding = umap_model.fit_transform(embeddings)
            joblib.dump(umap_embedding, umap_embedding_path)

            logging.info("Generating UMAP visualization...")
            draw_umap(
                umap_embedding,
                color_data=embeddings[:, 0],  # Or some label/cluster assignment for better color
                filename=umap_plot_path,
                title=f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"
            )

            run_elapsed_time = time.time() - run_start_time
            logging.info(f"Finished run in {run_elapsed_time:.2f} seconds. Results saved in '{run_output_dir}'")


        total_elapsed_time = time.time() - total_start_time
        logging.info(f"\nAll UMAP processing complete in {total_elapsed_time:.2f} seconds.")

    except Exception:
        logging.exception("An error occurred during execution.")
        sys.exit(1)


if __name__ == '__main__':
    main()
