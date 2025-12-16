import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, 
    roc_curve, auc, roc_auc_score, 
    precision_recall_fscore_support, matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from itertools import cycle
import argparse # Added for better command-line execution

# --- Configuration (Must match train_and_evaluate.py) ---
MODEL_NAME = 'DiCNN_Kmer_Classifier'
ARTIFACTS_DIR = f'./{MODEL_NAME}_artifacts'
MODEL_PATH = os.path.join(ARTIFACTS_DIR, f'{MODEL_NAME}_weights.h5')
CLASSES_PATH = os.path.join(ARTIFACTS_DIR, 'label_classes.json')

DATA_PATH = 'Flavi_training_data.csv' 
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
        return None, None, None, None

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
    kmer_data_config = {}
    
    for k in k_sizes:
        tokenizer_path = os.path.join(artifacts_dir, f'tokenizer_k{k}.json')
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_config)

        all_kmers = [generate_kmers(seq, k) for seq in dna_sequences]
        encoded_sequences = tokenizer.texts_to_sequences(all_kmers)
        
        max_kmer_seq_len = max(1, sequence_length - k + 1)
        padded_sequences = pad_sequences(encoded_sequences, maxlen=max_kmer_seq_len, padding='post')
        
        input_data[f'input_k{k}'] = padded_sequences
        kmer_data_config[k] = {'max_len': max_kmer_seq_len} # Store max_len for verification

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
        X_test_dict[f'input_k{k_size}'] = split_results[i * 2 + 1]

    y_test = split_results[num_input_features * 2 + 1]

    return X_test_dict, y_test, label_encoder, class_names

def calculate_metrics(y_true_classes, y_pred_classes, y_pred_proba, class_names):
    """Calculates and prints statistical classification metrics."""
    
    print("\n--- Statistical Classification Metrics ---")
    
    # 1. Precision, Recall, F1-Score (Per Class)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_classes, y_pred_classes, labels=np.arange(len(class_names)), zero_division=0
    )

    prf_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    print("\n🔍 Precision / Recall / F1-Score (Per Class):")
    print(prf_df.round(4).to_string(index=False))

    # 2. Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true_classes, y_pred_classes)
    print(f"\n📊 Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    # 3. AUC-ROC (macro/micro)
    try:
        # Check if probability output matches true classes (important for multi-class)
        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
            auc_macro = roc_auc_score(y_true_classes, y_pred_proba, multi_class="ovr", average="macro")
            auc_micro = roc_auc_score(y_true_classes, y_pred_proba, multi_class="ovr", average="micro")
        else:
            # Binary case or single-class output which is not correct for multi-class problem
            raise ValueError("y_pred_proba does not have correct shape for multi-class AUC.")

    except ValueError as e:
        auc_macro = auc_micro = np.nan
        print(f"\n⚠️ AUC-ROC Calculation Skipped: {e}")
        
    print(f"\n📈 AUC-ROC Macro: {auc_macro:.4f}")
    print(f"📈 AUC-ROC Micro: {auc_micro:.4f}")


def plot_roc_curve(y_true_classes, y_pred_proba, class_names, output_dir):
    """Computes and plots the ROC curve (per-class, micro, and macro average)."""
    
    # Binarize the true labels
    y_true_bin = label_binarize(y_true_classes, classes=np.arange(len(class_names)))
    n_classes = y_true_bin.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area (via interpolation)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    plt.figure(figsize=(12, 10))
    colors = cycle(plt.cm.tab10.colors)

    # Plot per class 
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    # Plot micro/macro averages
    plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=3,
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', linewidth=3,
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('Multi-Class AUC-ROC Curve', fontsize=18)
    plt.legend(loc="lower right", fontsize=10)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    # Save figures
    cm_filename = os.path.join(output_dir, 'DiCNN-UniK_curve')
    plt.savefig(f'{cm_filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{cm_filename}.png', dpi=300, bbox_inches='tight')
    
    print("\nAUC-ROC Curve generated and saved to artifacts folder.")
    # plt.show()


def plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, output_dir):
    """Computes and plots the Confusion Matrix."""
    
    print("\n--- Generating Confusion Matrix ---")
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm, values_format='d', xticks_rotation='vertical')
    
    ax_cm.set_title('DiCNN-UniK Test Set Confusion Matrix', fontsize=16)
    plt.xticks(fontsize=12) 
    plt.yticks(fontsize=12)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('True Class', fontsize=14)
    plt.tight_layout()

    # Save Figures
    cm_filename = os.path.join(output_dir, 'DiCNN-UniK_testcm')
    plt.savefig(f'{cm_filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{cm_filename}.png', dpi=300, bbox_inches='tight')
    print("Confusion Matrix generated and saved to artifacts folder.")
    # plt.show()


def main(data_path, k_sizes):
    """Main evaluation function."""
    
    if not os.path.exists(ARTIFACTS_DIR) or not os.path.exists(MODEL_PATH):
        print("ERROR: Model artifacts not found.")
        print("Please run 'python train_and_evaluate.py' first to generate the model and artifacts.")
        return

    # 1. Prepare data (re-split from original CSV)
    print("Loading data and artifacts to reconstruct test set...")
    X_test_dict, y_test, label_encoder, class_names = load_data_and_split(data_path, k_sizes, ARTIFACTS_DIR)

    if X_test_dict is None:
        return

    # 2. Load Model and Predict
    try:
        loaded_model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model from {MODEL_PATH}. Details: {e}")
        return

    print("Model loaded successfully. Starting prediction...")
    
    y_pred_proba = loaded_model.predict(X_test_dict, verbose=1)
    y_pred_classes = np.argmax(y_pred_proba, axis=1) # predicted class indices
    y_true_classes = np.argmax(y_test, axis=1) # true class indices

    # 3. Calculate and Print Metrics
    calculate_metrics(y_true_classes, y_pred_classes, y_pred_proba, class_names)

    # 4. Plot Confusion Matrix
    plot_confusion_matrix(y_true_classes, y_pred_classes, class_names, ARTIFACTS_DIR)
    
    # 5. Plot AUC-ROC Curve
    plot_roc_curve(y_true_classes, y_pred_proba, class_names, ARTIFACTS_DIR)
    
    print("\nEvaluation complete. Check the 'DiCNN_Kmer_Classifier_artifacts' folder for results.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the Multi-Kmer DiCNN-UniK model and generate metrics/plots.")
    
    parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the CSV training data file.')
    parser.add_argument('--k_sizes', type=int, nargs='+', default=K_SIZES, help='List of k-mer sizes to use (must match training).')

    args = parser.parse_args()
    
    main(args.data_path, args.k_sizes)
