import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --- Configuration (Must match train_and_evaluate.py) ---
MODEL_NAME = 'DiCNN_Kmer_Classifier'
ARTIFACTS_DIR = f'./{MODEL_NAME}_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, f'{MODEL_NAME}_weights.h5')
CLASSES_PATH = os.path.join(ARTIFACTS_DIR, 'label_classes.json')

DATA_PATH = 'Flavi_training_data.csv' # Assuming data is loaded/placed here for re-splitting
TEST_SIZE = 0.2
RANDOM_STATE = 5691
K_SIZES = [5, 6]


def generate_kmers(sequence, k):
    """Generates overlapping k-mers from a DNA sequence."""
    if len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def load_data_and_split(data_path, k_sizes, artifacts_dir):
    """
    Loads data, tokenizers, performs preprocessing, and splits into train/test sets
    to accurately reconstruct the X_test_dict and y_test used during training.
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please download it as per README.")
        return None, None, None

    dna_sequences = df['Sequence']
    labels = df['Organism_Name']
    sequence_length = dna_sequences.apply(len).max()

    # --- 1. Load Preprocessing Artifacts ---
    with open(CLASSES_PATH, 'r') as f:
        class_names = json.load(f)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(class_names) # Fit to the saved classes
    integer_encoded_labels = label_encoder.transform(labels)
    num_classes = len(class_names)
    labels_one_hot = to_categorical(integer_encoded_labels, num_classes=num_classes)

    # --- 2. Load Tokenizers and Process Data ---
    input_data = {}
    
    for k in k_sizes:
        # Load the tokenizer from JSON
        tokenizer_path = os.path.join(artifacts_dir, f'tokenizer_k{k}.json')
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_config)

        # Re-generate k-mers and encode/pad (must be the exact same process)
        all_kmers = [generate_kmers(seq, k) for seq in dna_sequences]
        encoded_sequences = tokenizer.texts_to_sequences(all_kmers)
        
        max_kmer_seq_len = max(1, sequence_length - k + 1)
        padded_sequences = pad_sequences(encoded_sequences, maxlen=max_kmer_seq_len, padding='post')
        
        input_data[f'input_k{k}'] = padded_sequences

    # --- 3. Re-split Data ---
    inputs_for_split = list(input_data.values()) + [labels_one_hot]
    split_results = train_test_split(
        *inputs_for_split, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

    X_test_dict = {}
    num_input_features = len(k_sizes)

    for i, k_size in enumerate(k_sizes):
        # The test set is the second entry in each pair of split results
        X_test_dict[f'input_k{k_size}'] = split_results[i * 2 + 1]

    # y_test is the last entry
    y_test = split_results[num_input_features * 2 + 1]

    return X_test_dict, y_test, label_encoder

def plot_confusion_matrix(X_test_dict, y_test, label_encoder, model_path, output_dir):
    """
    Loads the trained model, performs inference, and plots the confusion matrix.
    """
    print("\n--- Loading Model and Artifacts ---")
    try:
        # NOTE: If your model uses custom layers, you must pass them via custom_objects
        loaded_model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Did you train the model first?")
        print(f"Details: {e}")
        return

    print("Model loaded successfully. Starting prediction...")
    
    # --- 1. Predict and Decode ---
    y_pred_proba = loaded_model.predict(X_test_dict, verbose=1)
    y_pred_classes = np.argmax(y_pred_proba, axis=1) # get predicted class indices

    y_true_classes = np.argmax(y_test, axis=1) # Convert y_test from one-hot to class indices

    class_names = label_encoder.classes_.tolist()

    # --- 2. Compute and Display Confusion Matrix ---
    print("\n--- Generating Confusion Matrix ---")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Display confusion matrix 
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plotting details
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d', xticks_rotation='vertical')
    
    ax_cm.set_title('DiCNN-UniK Test Set Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.tight_layout()

    # --- 3. Save Figures ---
    os.makedirs(output_dir, exist_ok=True)
    
    cm_filename = os.path.join(output_dir, 'DiCNN-UniK_testcm')
    plt.savefig(f'{cm_filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{cm_filename}.png', dpi=300, bbox_inches='tight')
    print(f"Confusion Matrix saved to {output_dir}/ as PDF and PNG files.")
    
    # Optional: Display the matrix interactively (only if not running in headless mode)
    # plt.show()


if __name__ == '__main__':
    # Check for artifacts first
    if not os.path.exists(ARTIFACTS_DIR) or not os.path.exists(MODEL_PATH):
        print("ERROR: Model artifacts not found.")
        print("Please run 'python train_and_evaluate.py' first to generate the model and artifacts.")
    else:
        # 1. Prepare data (re-split from original CSV)
        X_test_dict, y_test, label_encoder = load_data_and_split(DATA_PATH, K_SIZES, ARTIFACTS_DIR)

        if X_test_dict is not None:
            # 2. Run evaluation and plotting
            plot_confusion_matrix(X_test_dict, y_test, label_encoder, MODEL_PATH, ARTIFACTS_DIR)
