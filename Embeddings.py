from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os
import configparser # Import the configparser module

# --- Step 0: Load Configuration ---
config = configparser.ConfigParser()
config_file_path = 'config.ini' # Define the name of your config file

try:
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"Error: Configuration file '{config_file_path}' not found.")
    config.read(config_file_path)

    # Get values from the config file
    json_file_path = config.get('Paths', 'json_file_path')
    save_file_name = config.get('Paths', 'save_file_name')
    embedding_model_name = config.get('Models', 'embedding_model_name')

    print("Configuration loaded successfully:")
    print(f"  JSON file path: {json_file_path}")
    print(f"  Save file name: {save_file_name}")
    print(f"  Embedding model: {embedding_model_name}")

except FileNotFoundError as e:
    print(e)
    exit()
except configparser.NoSectionError as e:
    print(f"Error in config file: {e}. Ensure sections like [Paths] and [Models] exist.")
    exit()
except configparser.NoOptionError as e:
    print(f"Error in config file: {e}. Ensure all required options are present.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading the configuration: {e}")
    exit()


# --- Step 1: Prepare your commit messages from a JSON file ---

commit_messages = [] # Initialize an empty list to store messages

try:
    print(f"\nLoading commit messages from {json_file_path}...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        # Load the entire JSON content
        commit_data = json.load(f)

    # Check if the loaded data is a list and if entries have the 'Message' key
    if isinstance(commit_data, list) and all(isinstance(entry, dict) and 'Message' in entry for entry in commit_data):
        # Extract the 'Message' from each dictionary in the list
        commit_messages = [entry['Message'].replace("-", " ").lower() for entry in commit_data]
        print(f"Successfully loaded {len(commit_messages)} commit messages.")
    else:
        print(f"Error: JSON data in {json_file_path} is not in the expected format (list of objects with 'Message' key).")
        exit()

except FileNotFoundError:
    print(f"Error: The file {json_file_path} was not found. Please ensure the path is correct.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {json_file_path}. Please check the file format.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading messages: {e}")
    exit()

if not commit_messages:
    print("No commit messages were loaded. Please check your JSON file and the loading logic.")
    exit()


# --- Step 2: Load Embedding Model and Pre-calculate Embeddings ---
print(f"\nLoading embedding model: {embedding_model_name}...")
try:
    embedding_model = SentenceTransformer(embedding_model_name)
    print("Embedding model loaded successfully.")
    print("\nCalculating embeddings (this may take a while)...")
    embeddings = embedding_model.encode(commit_messages, show_progress_bar=True)
except Exception as e:
    print(f"An error occurred during embedding model loading or encoding: {e}")
    exit()

# --- Step 3: Save Embeddings ---
try:
    # Ensure the directory for the save file exists if a path is specified
    save_file_dir = os.path.dirname(save_file_name)
    if save_file_dir and not os.path.exists(save_file_dir):
        os.makedirs(save_file_dir)
        print(f"Created directory: {save_file_dir}")

    np.save(save_file_name, embeddings)
    print(f"\nEmbeddings successfully saved to {save_file_name}.")
except Exception as e:
    print(f"An error occurred while saving embeddings: {e}")
    exit()

print("\nProcessing complete.")
