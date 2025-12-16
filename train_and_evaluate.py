import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from model_architecture import build_multi_kmer_cnn # Import the model function
import argparse
import os
import json # To save tokenizer data

# --- Configuration ---
DEFAULT_K_SIZES = [5, 6]
MODEL_NAME = 'DiCNN_Kmer_Classifier'
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 5691
DATA_PATH = 'Flavi_training_data.csv'


def generate_kmers(sequence, k):
    """Generates overlapping k-mers from a DNA sequence."""
    if len(sequence) < k:
        return []
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def preprocess_data(df, k_sizes, sequence_length):
    """Generates k-mer data and tokenizes sequences for all k-mer sizes."""
    kmer_data_config = {}
    input_data = {}

    for k in k_sizes:
        print(f"\n--- Processing k-mer size: {k} ---")
        
        # 1. Generate K-mers
        all_kmers = [generate_kmers(seq, k) for seq in df['Sequence']]

        # 2. Tokenize
        tokenizer = Tokenizer(char_level=False, filters='', lower=False)
        tokenizer.fit_on_texts(all_kmers)

        # 3. Encode and Pad Sequences
        encoded_sequences = tokenizer.texts_to_sequences(all_kmers)

        # The length of the k-mer sequence is the original sequence length minus k + 1
        max_kmer_seq_len = max(1, sequence_length - k + 1)
        
        padded_sequences = pad_sequences(encoded_sequences, maxlen=max_kmer_seq_len, padding='post')

        # 4. Store configuration and data
        vocab_size = len(tokenizer.word_index) + 1
        
        kmer_data_config[k] = {
            'vocab_size': vocab_size,
            'max_len': max_kmer_seq_len,
            # NOTE: We save the tokenizer separately later, but store it here temporarily
            'tokenizer': tokenizer 
        }
        input_data[f'input_k{k}'] = padded_sequences
        
        print(f"  Vocab size: {vocab_size}")
        print(f"  Max sequence length: {max_kmer_seq_len}")
        print(f"  Padded data shape: {padded_sequences.shape}")
        
    return input_data, kmer_data_config

def main(data_path, k_sizes, epochs, batch_size, model_name):
    """Main training and evaluation function."""
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    
    dna_sequences = df['Sequence']
    labels = df['Organism_Name']
    sequence_length = dna_sequences.apply(len).max()

    print(f"Data loaded successfully. Total sequences: {len(df)}")
    print(f"Max sequence length detected: {sequence_length}")

    # --- 2. Label Encoding and One-Hot Encoding for Labels ---
    label_encoder = LabelEncoder()
    integer_encoded_labels = label_encoder.fit_transform(labels)
    num_classes = len(label_encoder.classes_)
    labels_one_hot = to_categorical(integer_encoded_labels, num_classes=num_classes)
    
    print(f"Detected {num_classes} unique classes: {label_encoder.classes_.tolist()}")

    # --- 3. K-mer Generation and Tokenization ---
    input_data, kmer_data_config = preprocess_data(df, k_sizes, sequence_length)

    # --- 4. Data Splitting ---
    inputs_for_split = list(input_data.values()) + [labels_one_hot]
    split_results = train_test_split(
        *inputs_for_split, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )

    X_train_dict = {}
    X_test_dict = {}
    num_input_features = len(k_sizes)

    # Distribute split results back into training/testing dictionaries
    for i, k_size in enumerate(k_sizes):
        X_train_dict[f'input_k{k_size}'] = split_results[i * 2]
        X_test_dict[f'input_k{k_size}'] = split_results[i * 2 + 1]

    y_train = split_results[num_input_features * 2]
    y_test = split_results[num_input_features * 2 + 1]

    print("\nData splitting complete.")
    print(f"Training data shape: {y_train.shape}")
    print(f"Test data shape: {y_test.shape}")
    
    # --- 5. Build and Compile Model ---
    # Prepare the config to pass to the model builder (removing temporary tokenizer object)
    model_config = {k: {key: val for key, val in v.items() if key != 'tokenizer'} 
                    for k, v in kmer_data_config.items()}

    model = build_multi_kmer_cnn(model_config, num_classes)
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()

    # --- 6. Train the Model ---
    print("\nStarting model training...")
    history = model.fit(
        X_train_dict,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_dict, y_test),
        verbose=1
    )

    print("\nModel training complete.")

    # --- 7. Evaluate and Save Results ---
    loss, accuracy = model.evaluate(X_test_dict, y_test, verbose=0)
    print("---------------------------------------")
    print(f"Final Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print("---------------------------------------")

    # Create a directory to save artifacts
    save_dir = f'./{model_name}_artifacts'
    os.makedirs(save_dir, exist_ok=True)
    
    # Save Model Weights
    model.save(os.path.join(save_dir, f'{model_name}_weights.h5'))
    print(f"Model saved to {os.path.join(save_dir, f'{model_name}_weights.h5')}")

    # Save Label Encoder Classes (for inference/deployment)
    with open(os.path.join(save_dir, 'label_classes.json'), 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f)
    
    # Save Tokenizers (CRITICAL for inference)
    for k, config in kmer_data_config.items():
        tokenizer = config['tokenizer']
        # Keras tokenizer needs to be saved to disk
        tokenizer_json = tokenizer.to_json()
        with open(os.path.join(save_dir, f'tokenizer_k{k}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(tokenizer_json, ensure_ascii=False))
        
    print(f"Preprocessing artifacts saved to {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the Multi-Kmer DiCNN-UniK model for Flavivirus classification.")
    
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the CSV training data file.')
    parser.add_argument('--k_sizes', type=int, nargs='+', default=DEFAULT_K_SIZES, help='List of k-mer sizes to use (e.g., 5 6).')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Training batch size.')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME, help='Name for the saved model files.')

    args = parser.parse_args()
    
    main(args.data_path, args.k_sizes, args.epochs, args.batch_size, args.model_name)
