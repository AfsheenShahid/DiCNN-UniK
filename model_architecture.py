import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Dropout

def build_multi_kmer_cnn(kmer_data_config, num_classes, embedding_dim=128, filters_per_conv=256, kernel_sizes_conv=[3, 5]):
    """
    Builds a Multi-Kmer Input CNN model based on the provided k-mer configurations.

    Args:
        kmer_data_config (dict): A dictionary where keys are k-mer sizes (e.g., 5, 6) 
                                 and values are dictionaries containing 'vocab_size' 
                                 and 'max_len' for that k-mer.
        num_classes (int): The number of output classes.
        embedding_dim (int): The dimension of the k-mer embedding space.
        filters_per_conv (int): Number of filters for each Conv1D layer.
        kernel_sizes_conv (list): List of kernel sizes for the Conv1D layers.

    Returns:
        tf.keras.Model: The compiled multi-input CNN model.
    """
    
    #  
    # This architecture uses multiple parallel branches (one for each k-mer size)
    # before merging and classification, similar to the abstract's DiCNN concept.
    
    input_branches = []
    pooled_features = []

    # ------------------ K-mer Specific Branches ------------------
    for k_size, config in kmer_data_config.items():
        k = k_size
        max_len = config['max_len']
        vocab_size = config['vocab_size']
        
        # 1. Input Layer
        input_layer = Input(shape=(max_len,), name=f'input_k{k}')
        input_branches.append(input_layer)

        # 2. Embedding Layer
        embedding_layer = Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_len,
            name=f'embedding_k{k}'
        )(input_layer)

        # 3. Parallel 1D Convolutional Layers (for different kernel sizes)
        conv_outputs = []
        for kernel_size in kernel_sizes_conv:
            conv = Conv1D(
                filters=filters_per_conv, 
                kernel_size=kernel_size, 
                activation='relu',
                name=f'conv1d_k{k}_ks{kernel_size}'
            )(embedding_layer)
            
            # GlobalMaxPooling1D pools features across the entire sequence length
            conv_outputs.append(GlobalMaxPooling1D()(conv))

        # 4. Merge Convolutional Features (if multiple kernels are used)
        if len(conv_outputs) > 1:
            merged_conv = concatenate(conv_outputs, name=f'merged_conv_k{k}')
        else:
            merged_conv = conv_outputs[0]

        pooled_features.append(merged_conv)

    # ------------------ Global Merging and Classification Head ------------------
    if len(pooled_features) > 1:
        all_features_merged = concatenate(pooled_features, name='all_kmer_features_merged')
    else:
        all_features_merged = pooled_features[0]

    # Classification Head (Dense Layers with Dropout)
    dense_layer_1 = Dense(512, activation='relu')(all_features_merged)
    dropout_layer_1 = Dropout(0.5)(dense_layer_1)

    dense_layer_2 = Dense(256, activation='relu')(dropout_layer_1)
    dropout_layer_2 = Dropout(0.5)(dense_layer_2)

    output_layer = Dense(num_classes, activation='softmax', name='output_classification')(dropout_layer_2)

    # Final Model Definition
    model = Model(inputs=input_branches, outputs=output_layer, name='DNA_Kmer_Flavivirus_Classifier')

    return model

if __name__ == '__main__':
    # This block is for testing the model architecture independently
    print("Testing model build function with dummy data...")
    
    # Dummy configuration
    dummy_config = {
        5: {'vocab_size': 10000, 'max_len': 500},
        6: {'vocab_size': 40000, 'max_len': 499}
    }
    dummy_num_classes = 10

    test_model = build_multi_kmer_cnn(dummy_config, dummy_num_classes)
    test_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    test_model.summary()
    print("Model built successfully.")
