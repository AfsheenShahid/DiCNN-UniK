import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Dropout
from tensorflow.keras.utils import model_to_dot
import pydot
import os

# Updated Parameters ---
num_classes = 10
embedding_dim = 128
filters_per_conv = 256
kernel_sizes_conv = [3, 5]

# New Derived Parameters
kmer_config = {
    5: {
        'vocab_size': 1025,   # Derived from Param count (131200 / 128)
        'max_len': 11560      # Matches input_k5 Output Shape
    },
    6: {
        'vocab_size': 4097,   # Derived from Param count (524416 / 128)
        'max_len': 11559      # Matches input_k6 Output Shape
    }
}

def build_model(kmer_config, k_sizes, num_classes, embedding_dim, filters_per_conv, kernel_sizes_conv):
    """Builds the updated DiCNN-UniK architecture."""
    input_branches = []
    pooled_features = []

    for k in k_sizes:
        max_kmer_len = kmer_config[k]['max_len']
        vocab_size = kmer_config[k]['vocab_size']

        input_layer = Input(shape=(max_kmer_len,), name=f'input_k{k}')
        input_branches.append(input_layer)

        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name=f'embedding_{k}' if k == 6 else 'embedding' # Matches your table names
        )(input_layer)

        conv_outputs = []
        for i, kernel_size in enumerate(kernel_sizes_conv):
            # Naming conv layers to match: conv1d, conv1d_1, conv1d_2, conv1d_3
            suffix = f"_{len(pooled_features)*2 + i}" if (len(pooled_features)*2 + i) > 0 else ""
            conv = Conv1D(
                filters=filters_per_conv,
                kernel_size=kernel_size,
                activation='relu',
                name=f'conv1d{suffix}'
            )(embedding_layer)

            pool = GlobalMaxPooling1D(name=f'global_max_pooling1d{suffix}')(conv)
            conv_outputs.append(pool)

        merged_conv = concatenate(conv_outputs, name=f'merged_conv_k{k}')
        pooled_features.append(merged_conv)

    all_features_merged = concatenate(pooled_features, name='all_kmer_features_merged')

    x = Dense(512, activation='relu', name='dense')(all_features_merged)
    x = Dropout(0.5, name='dropout')(x)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(0.5, name='dropout_1')(x)

    output_layer = Dense(num_classes, activation='softmax', name='output_classification')(x)

    model = Model(inputs=input_branches, outputs=output_layer, name='DiCNN_UniK_Updated')
    return model

def plot_model_with_custom_style(model, file_name, node_color='lightgray', font_size='20'):
    """
    Saves model plot with custom colors AND increased font size.
    """
    try:
        dot = model_to_dot(
            model,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=300 # Higher resolution
        )

        # Update Node styles
        for node in dot.get_nodes():
            node.set('fillcolor', node_color)
            node.set('style', 'filled')
            node.set('fontsize', font_size) # <--- INCREASES FONT SIZE
            node.set('fontname', 'Arial Bold')

        # Save files
        if file_name.endswith('.pdf'): dot.write_pdf(file_name)
        elif file_name.endswith('.png'): dot.write_png(file_name)
        elif file_name.endswith('.eps'): dot.write_postscript(file_name)

        print(f"✅ Saved architecture to: {file_name} (Font Size: {font_size})")

    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")

# Execution ---
if __name__ == '__main__':
    model = build_model(kmer_config, [5, 6], num_classes, embedding_dim, filters_per_conv, kernel_sizes_conv)

    # Show summary to verify it matches your table
    model.summary()

    # Generate the high-visibility plots
    plot_model_with_custom_style(model, 'DiCNN_architecture_high_res.png', font_size='24')
    plot_model_with_custom_style(model, 'DiCNN_architecture_high_res.pdf', font_size='24')
