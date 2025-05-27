import configparser
import json
import os

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# --- Step 0: Load Configuration ---
config = configparser.ConfigParser()
config_file_path = 'config.ini'

try:
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Error: Configuration file '{config_file_path}' not found.")
    config.read(config_file_path)

    # Paths
    json_file_path = config.get('Paths', 'json_file_path')
    embeddings_input_path = config.get('Paths', 'embeddings_input_path') # For loading pre-calculated embeddings
    results_output_dir = config.get('Paths', 'results_output_dir')

    # Models
    embedding_model_name = config.get('Models', 'embedding_model_name')

    # BERTopic Parameters
    ngram_lower = config.getint('BERTopicParams', 'ngram_range_lower')
    ngram_upper = config.getint('BERTopicParams', 'ngram_range_upper')
    mmr_diversity_param = config.getfloat('BERTopicParams', 'mmr_diversity')
    min_cluster_size_param = config.getint('BERTopicParams', 'hdbscan_min_cluster_size')

    print("✅ Configuration loaded successfully:")
    print(f"  JSON file path: {json_file_path}")
    print(f"  Embeddings input path: {embeddings_input_path}")
    print(f"  Results output directory: {results_output_dir}")
    print(f"  Embedding model: {embedding_model_name}")
    print(f"  N-gram range: ({ngram_lower}, {ngram_upper})")
    print(f"  MMR diversity: {mmr_diversity_param}")
    print(f"  HDBSCAN min_cluster_size: {min_cluster_size_param}")

except FileNotFoundError as e:
    print(f"❌ {e}")
    exit()
except configparser.NoSectionError as e:
    print(f"❌ Error in config file: {e}. Ensure all sections like [Paths], [Models], [BERTopicParams] exist.")
    exit()
except configparser.NoOptionError as e:
    print(f"❌ Error in config file: {e}. Ensure all required options are present.")
    exit()
except ValueError as e:
    print(f"❌ Error in config file: A parameter has an incorrect type (e.g., expected a number but got text): {e}")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while loading the configuration: {e}")
    exit()

# --- Step 1: Prepare your commit messages from a JSON file ---
commit_messages = []
try:
    print(f"\n🔄 Loading commit messages from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        commit_data = json.load(f)
    if isinstance(commit_data, list) and all(isinstance(entry, dict) and 'Message' in entry for entry in commit_data):
        commit_messages = [entry['Message'].replace("-", " ").lower() for entry in commit_data]
        print(f"👍 Successfully loaded {len(commit_messages)} commit messages.")
    else:
        print(f"❌ Error: JSON data in {json_file_path} is not in the expected format.")
        exit()
except FileNotFoundError:
    print(f"❌ Error: The file {json_file_path} was not found.")
    exit()
except json.JSONDecodeError:
    print(f"❌ Error: Could not decode JSON from {json_file_path}.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while loading messages: {e}")
    exit()

if not commit_messages:
    print("⚠️ No commit messages were loaded. Please check your JSON file.")
    exit()

# --- Step 2: Load Pre-calculated Embeddings ---
print(f"\n🔄 Loading pre-calculated embeddings from {embeddings_input_path}...")
try:
    embeddings = np.load(embeddings_input_path)
    print(f"👍 Embeddings loaded successfully. Shape: {embeddings.shape}")
except FileNotFoundError:
    print(f"❌ Error: Embeddings file '{embeddings_input_path}' not found. Please ensure you have run the embedding generation script first.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading embeddings: {e}")
    exit()


# --- Step 3: Initialize Models with Configured Parameters ---
print("\n⚙️ Initializing models with configured parameters...")

# Embedding model (used by BERTopic if not passing pre-computed embeddings directly for fit_transform,
# or for other internal uses like topic reduction or visualization if specified)
# If you only use pre-computed embeddings for fit_transform and don't need the model object for other BERTopic features,
# you could potentially make loading this optional or pass its name directly to BERTopic's save_embedding_model.
# For now, we load it as it's good practice for full BERTopic functionality.
try:
    embedding_model = SentenceTransformer(embedding_model_name)
    print(f"  Embedding model '{embedding_model_name}' loaded.")
except Exception as e:
    print(f"❌ Error loading SentenceTransformer model '{embedding_model_name}': {e}")
    # Decide if you want to exit or try to proceed without it for some operations
    # For this script, BERTopic will likely need it or its name.
    exit()


# Attempt to import cuML components
try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    use_cuml = True
    print("  cuML found. Will attempt to use GPU for UMAP and HDBSCAN.")
except ImportError:
    use_cuml = False
    print("  cuML not found. UMAP and HDBSCAN will run on CPU.")
    from umap import UMAP
    from hdbscan import HDBSCAN


# Define UMAP and HDBSCAN models
if use_cuml:
    # TODO: Make UMAP params configurable if needed via config.ini
    umap_model = cumlUMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42)
    hdbscan_model = cumlHDBSCAN(min_cluster_size=min_cluster_size_param, # From config
                                min_samples=1, # Example: make configurable if needed
                                metric='euclidean',
                                gen_min_span_tree=True,
                                prediction_data=True)
else:
    # TODO: Make UMAP params configurable if needed via config.ini
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size_param, # From config
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True)
print(f"  HDBSCAN min_cluster_size set to: {min_cluster_size_param}")

# Representation Models
mmr = MaximalMarginalRelevance(diversity=mmr_diversity_param) # From config
keybert = KeyBERTInspired()
representation_model = [mmr, keybert]
print(f"  MMR diversity set to: {mmr_diversity_param}")

# Vectorizer and CTF-IDF
vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(ngram_lower, ngram_upper)) # From config
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) # reduce_frequent_words could also be in config
print(f"  Vectorizer N-gram range set to: ({ngram_lower}, {ngram_upper})")

# BERTopic Model
topic_model = BERTopic(
    vectorizer_model=vectorizer_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True,
    embedding_model=embedding_model, # Pass the loaded model object
    representation_model=representation_model,
    ctfidf_model=ctfidf_model,
    nr_topics="auto", # This could also be made configurable
)
print("👍 Models initialized.")

# --- Step 4: Fit the BERTopic model ---
print("\n🔄 Fitting BERTopic model...")
try:
    topics, probs = topic_model.fit_transform(commit_messages, embeddings)
    print("👍 Model fitting complete.")
except Exception as e:
    print(f"❌ An error occurred during BERTopic model fitting: {e}")
    exit()

# --- Step 5: Get and display topic information ---
print("\n📊 Discovered Topics:")
try:
    most_frequent_topics = topic_model.get_topic_info()
    print(most_frequent_topics)
    print(f"\nTotal number of topics found (including outliers): {len(topic_model.get_topics())}")
    print(f"Number of outlier messages (-1 topic): {list(topics).count(-1)}")
except Exception as e:
    print(f"❌ An error occurred while getting topic information: {e}")


# --- Step 6: Save the results ---
print(f"\n💾 Saving results to '{results_output_dir}/'...")
try:
    os.makedirs(results_output_dir, exist_ok=True) # Use configured output directory

    model_save_path = os.path.join(results_output_dir, "bertopic_model")
    # When saving, pass the name of the embedding model if you want it to be part of the saved BERTopic model config
    # and you are using a string reference like "BAAI/bge-large-en-v1.5".
    # If you passed a SentenceTransformer object to BERTopic's embedding_model parameter,
    # BERTopic might handle it differently or you might not need to specify save_embedding_model here
    # if the model object itself is being serialized (though typically it's the reference).
    topic_model.save(model_save_path, serialization="safetensors", save_embedding_model=embedding_model_name)
    print(f"  BERTopic model saved to {model_save_path}")

    results_df = pd.DataFrame({'Message': commit_messages, 'Topic': topics})
    assignments_save_path = os.path.join(results_output_dir, "commit_topic_assignments.csv")
    results_df.to_csv(assignments_save_path, index=False)
    print(f"  Topic assignments saved to {assignments_save_path}")

    if 'most_frequent_topics' in locals(): # Check if dataframe was created
        topic_info_save_path = os.path.join(results_output_dir, "topic_information.csv")
        most_frequent_topics.to_csv(topic_info_save_path, index=False)
        print(f"  Topic information summary saved to {topic_info_save_path}")

    # The following line is redundant if you already saved with topic_model.save() above
    # and specified save_embedding_model. Safetensors usually bundles components.
    # If you need to save ctfidf separately for some reason, you can.
    # topic_model.save(results_output_dir, serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model_name)

    print("👍 Saving complete.")
except Exception as e:
    print(f"❌ An error occurred while saving results: {e}")

print("\n🎉 Processing finished.")
