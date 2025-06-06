import os
import json
import numpy as np
import configparser
import logging
import sys
from sentence_transformers import SentenceTransformer

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

# --- Utility Functions ---
def load_config(config_path='config.ini'):
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")

    config.read(config_path)

    try:
        json_path = config.get('Paths', 'json_file_path')
        save_path = config.get('Paths', 'embeddings_path')
        model_name = config.get('Models', 'embedding_model_name')
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        raise ValueError(f"Error in config file: {e}")

    logging.info("Configuration loaded successfully")
    return json_path, save_path, model_name


def load_commit_messages(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file '{json_path}' not found.")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(entry, dict) and 'Message' in entry for entry in data):
        raise ValueError("JSON data must be a list of objects with a 'Message' key.")

    messages = [entry['Message'].replace('-', ' ').lower() for entry in data]
    logging.info(f"Loaded {len(messages)} commit messages")
    return messages


def calculate_embeddings(model_name, messages):
    logging.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logging.info("Calculating embeddings (this may take a while)...")
    return model.encode(messages, show_progress_bar=True)


def save_embeddings(save_path, embeddings):
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)
    logging.info(f"Embeddings saved to '{save_path}'")


# --- Main Pipeline ---
def main():
    # The first argument is always the script name
    if len(sys.argv) > 1:
        print("First argument:", sys.argv[1])
    else:
        print("No arguments provided.")
        exit(1)
    try:
        json_path, save_path, model_name = load_config(sys.argv[1])
        messages = load_commit_messages(json_path)
        embeddings = calculate_embeddings(model_name, messages)
        save_embeddings(save_path, embeddings)
        logging.info("Processing complete.")
    except Exception as e:
        logging.error(f"{e}")
        exit(1)


if __name__ == '__main__':
    main()
