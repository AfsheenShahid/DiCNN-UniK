
# 🦠 DiCNN-UniK: Dual-Input CNN  based on Universal and Unique K-mer librariesfor Viral Sequence Classification

## Accurate Flavivirus Classification via Unique and Universal k-mer Embeddings



## 🚀 Simple Overview

**DiCNN-UniK** is a novel deep learning model for **highly accurate and robust classification of viral sequences**, specifically focused on **Flaviviruses**.

We developed this Dual-Input CNN (DiCNN) to overcome the major limitations of traditional bioinformatics methods (like Multiple Sequence Alignment, MSA) which struggle with incomplete or poor-quality genomic data.

### Key Highlights:
* **UniK Advantage:** We generate novel, direct embeddings using **unique and universal k-mer libraries**, providing a clear "picture" of the local genomic context instead of relying on simple frequency counts.
* **Full Genome Support:** The model is trained and runs on **full-length genomic sequences**, bypassing the restrictive length limits of many modern models.
* **Superior Performance:** Achieves an impressive **99% accuracy** on Flavivirus classification.
* **Real-World Ready:** Highly robust and reliable, maintaining high performance at **95% accuracy** even with genomic coverage as low as **20%**.

---
...
## ⚙️ Getting Started

### 1. Data Preparation (Handling Large Genomic Files)

The primary training dataset (`Flavi_training_data.csv`) is too large for GitHub and is hosted externally on Google Drive.

1.  **Download the Training Data:**
    Please click the direct link below to download the primary training dataset:
    
    [**DOWNLOAD: Flavi_training_data.csv (Google Drive)**](https://drive.google.com/uc?export=download&id=1tZRuUj9Nb8UDvuUczvxMUnK9irGo9PJ8)
    
    > **Note:** This link is configured to initiate a direct download and bypass the Google Drive preview.

2.  **Place the Data:**
    After downloading, you **must** place the `Flavi_training_data.csv` file inside a new folder named `data/` in the root of this repository.

    Your repository structure should look like this:

    ```
    .
    ├── data/
    │   └── Flavi_training_data.csv  <-- Place the downloaded file here
    ├── model_architecture.py
    ├── train_and_evaluate.py
    └── README.md
    ```

3.  **Run Training:**
    Once the data is in place, you can proceed with installation and training using the command line arguments:

    ```bash
    python train_and_evaluate.py --data_path data/Flavi_training_data.csv
    ```
