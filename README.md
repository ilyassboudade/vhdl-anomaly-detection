# 🚀 VHDL Anomaly Detector

> A manual Recurrent Neural Network (RNN) implementation that analyzes tokenized VHDL source code to detect syntactic and semantic errors prior to simulation. By parsing VHDL descriptions at the character/token level, the system leverages a recurrent architecture to retain long-range context and identify corrupted hardware code.

## 📸 Demo

![Project Demo / Dashboard](docs/assets/demo.png)*

## 🛠️ Tech Stack & Architecture

- **Core Language:** Python 3.10

- **Data:** HuggingFace Datasets

- **Deep Learning:** NumPy, Pygments (`VhdlLexer`)

- **Storage & Databases:** JSON, NumPy NPZ binary format, CSV

- **Tools & Environment:** Git, Jupyter Notebook, Linux/Bash

### System Architecture

1. **Ingestion & Data Preparation:** Extraction from HuggingFace dataset (`hdl2v/vhdl-dataset`), error-injection mutation strategy for generating balanced healthy (`label = 0`) and corrupted (`label = 1`) code profiles.

2. **Preprocessing & Tokenization:** Lexical analysis via `pygments.lexers.VhdlLexer`, case-normalization via `str.upper()`, vocabulary mapping, and padding/truncation to fixed input sizes.

3. **Modeling & Evaluation:** Sequential RNN architecture processing tokens to detect sequence-level anomalies and multi-class error types.

## ⚙️ Quickstart & Local Installation

Bash

```
# 1. Clone the repository
git clone https://github.com/your-username/vhdl-anomaly-detection.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Execute application / Run Notebook
jupyter notebook
```

## 📌 Key Features & Capabilities

- **Custom Synthetic Mutation Strategy:** Automated pipeline converting clean VHDL code into labeled anomaly profiles across multiple error types.

- **Custom Tokenization & Normalization:** Robust VHDL lexer integration with upper-case normalization and special sequence tokens (`<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`).

- **Character & Token-Level Sequential RNN:** Custom manual RNN handling long-range structural dependencies inherent in hardware description languages.

- **Structured Data Deliverables:** Pre-packaged dataset splits (`train.npz`, `val.npz`, `test.npz`) alongside vocabulary mappings (`vocab.json`, `error_types.json`).

# 

## 📦 Data Deliverables (`/data` directory)

| **Deliverable Artifact** | **Type**        | **Internal Schema / Purpose**                                                       |
| ------------------------ | --------------- | ----------------------------------------------------------------------------------- |
| **`vocab.json`**         | JSON Dictionary | Mapping `token_string ➔ integer_index` for embedding initialization.                |
| **`error_types.json`**   | JSON Dictionary | Mapping `error_string ➔ integer_index` for multi-class categorization.              |
| **`train.npz`**          | NumPy Binary    | Contains array tensors `input_ids (N, 512)`, `labels (N,)`, and `error_types (N,)`. |
| **`val.npz`**            | NumPy Binary    | Identical structure; dedicated to evaluation metrics and hyperparameter tuning.     |
| **`test.npz`**           | NumPy Binary    | **Held-out test dataset.** Kept locked until final evaluation.                      |
| **`vhdl_dataset.csv`**   | CSV Data        | Raw text of source strings, mutations, and corresponding labels for auditing.       |
