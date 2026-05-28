# 🛡️ VHDL Anomaly Detection (Logic Error Predictor)

> A deep learning pipeline that reads tokenized VHDL source code to detect structural and semantic logic errors before simulation, identifying the specific anomaly type via a Bidirectional LSTM with Attention.

---

## 📌 Project Overview & Architecture

This project treats VHDL source code as strictly sequential text data. Because logic errors often span across multiple lines and rely heavily on long-range context, the system uses a recurrent architecture to memorize and evaluate the state of the hardware description.

```
Token Embedding (128d) ➔ BiLSTM (Layer 1) ➔ BiLSTM (Layer 2) ➔ Attention Layer ➔ Dense + Sigmoid ➔ Binary Error Label + Multi-Class Error Type
```

### ⚙️ Key Hyperparameters

* **Vocab Size:** ~5,000 tokens (built using a frequency threshold of `MIN_FREQ=2`)
* **Embedding Dimension:** `128` (Sweeps planned for `64`, `128`, and `256`)
* **Hidden Dimension:** `256` per direction (`512` effective for BiLSTM)
* **Sequence Constraints:** `MAX_SEQ_LEN = 512` tokens (covers the ~95th percentile of code lengths)
* **Training Batch Size:** `64` (~700 MB VRAM footprint, highly optimized for an NVIDIA RTX 3060 12GB GPU)

---

## 👥 Team Structure & Workflow

The pipeline is built on a clear data handoff contract. Systems are designed to be decoupled, invisible, and resilient.


| Role                                       | Scope & Responsibilities                                                                                                                                                     | Key Deliverables                                                                               |
| :------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- |
| **Data Engineer** <br>*(Member 1 / User)*  | Dataset ingestion, preprocessing, normalization, AI-assisted mutation engineering, GHDL verification, tokenization, serialization, and stratified splitting.                 | `vocab.json`<br>`error_types.json`<br>`train.npz`, `val.npz`, `test.npz`<br>`vhdl_labeled.csv` |
| **ML Engineer** <br>*(Student 1)*          | Consumes encoded data pipelines via`tf.data`, optimizes embedding layers, constructs the dual-layer BiLSTM + Custom Attention layer architecture, and manages training logs. | Model weights<br>Training logs<br>Architecture report                                          |
| **Training & Eval Lead** <br>*(Student 2)* | Performs hyperparameter sweeps designed by Student 1, documents training dynamics, runs validation metrics via GHDL, and develops the final CLI integration tool.            | Evaluation metrics<br>CLI tool<br>Final project documentation                                  |

---

## 📊 Dataset & Mutation Strategy

* **Base Corpus:** `hdl2v/vhdl-dataset` on HuggingFace (`8,626` raw sample pairs).
* **The Challenge:** The source dataset is designed for VHDL-to-Verilog translation. It contains no pre-labeled errors.
* **The Solution:** To train a supervised binary and multi-class classifier, synthetic anomalies injection is needed:
  1. Parse clean, compilable VHDL code (`label = 0`, `error_type = "NONE"`).
  2. Introduce **exactly one subtle logic error**.
  3. Label the mutated code (`label = 1`) and classify it into one of the designated error types.

### Classified Error Types (`error_types.json`)

#### 1. Inverted Assignment

* **Target:** Data assignment expressions (typically signal assignments like `<=`).
* **Concept:** It simulates an accidental inversion of control or data polarity by wrapping the right-hand side (RHS) of an assignment statement with a logical `NOT`.
* **VHDL Example:**
* *Original:* `Q <= D;`
* *Mutated:* `Q <= NOT (D);`
* **Use Case:** Excellent for catching bit-flipping anomalies or tracking downstream data path propagation errors. It targets active signal updates rather than conditions.

#### 2. Relational Swap

* **Target:** Relational operators within conditional structures (e.g., statements inside an `IF`, `ELSIF`, or `WHEN` clause).
* **Concept:** It alters the boundary conditions of a comparison by swapping a relational operator (like `<`, `>`, `<=`, `>=`, `=`, `/=`) with its inverse or a closely related counterpart.
* **VHDL Example:**
* *Original:* `IF (COUNTER >= 10) THEN`
* *Mutated:* `IF (COUNTER > 10) THEN` *(or `COUNTER < 10` depending on the map)*
* **Use Case:** Simulates "off-by-one" bugs, boundary threshold errors, and state machine transitions that trigger too early, too late, or under the wrong comparative state.

#### 3. Logical Swap

* **Target:** Boolean logical operators (`AND`, `OR`, `XOR`) and edge-detection functions (`RISING_EDGE`, `FALLING_EDGE`).
* **Concept:** It alters the underlying combinatorial logic gating or clocking dependency by swapping operators within an expression.
* **VHDL Example:**
* *Original:* `IF (CLK'EVENT AND CLK = '1') AND RESET = '0' THEN`
* *Mutated:* `IF (CLK'EVENT AND CLK = '1') OR RESET = '0' THEN`
* *Another Example:* Swapping `RISING_EDGE(CLK)` to `FALLING_EDGE(CLK)`.
* **Use Case:** Crucial for verifying the robustness of combinatorial logic paths and clock-domain safety. It forces sequential blocks to trigger on inverted clock phases or corrupts complex multi-condition validation gates.


| Feature                   | Inverted Assignment                          | Relational Swap                          | Logical Swap                                                     |
| --------------------------- | ---------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------ |
| **Primary Code Focus**    | Data/Signal Assignment Lines                 | Conditional Expressions (`IF`, `WHEN`)   | Combinatorial/Clock Logic Expressions                            |
| **Operators Affected**    | `<=`, `:=`                                   | `>=`, `<=`, `>`, `<`, `=`, `/=`          | `AND`, `OR`, `XOR`, `XNOR`, `RISING_EDGE`, `FALLING_EDGE`        |
| **Modification Type**     | Insertion (`NOT (...)`)                      | One-to-one mapping replacement           | Regex word boundary string replacement                           |
| **Simulated Human Error** | Wrong polarity, incorrect gate output choice | Boundary value mistake, wrong comparison | Incorrect logic gate implementation, wrong clock phase selection |

---

## 🧼 Tokenization & Normalization Specifications

To ensure minimal vocabulary bloat and maximize embedding semantic strength:

* **Lexer:** `pygments.lexers.VhdlLexer`
* **Case Insensitivity:** All tokens are strictly uppercased via `str.upper()` to match VHDL specifications.
* **Special Tokens:**
  * `<PAD>` (`0`): Sequence padding element.
  * `<UNK>` (`1`): Out-of-vocabulary terms caught at inference.
  * `<BOS>` (`2`): Beginning of sequence indicator.
  * `<EOS>` (`3`): End of sequence indicator.

---

## 📈 Project Progress Tracker

### Completed

* [X] Ingested raw source data (`8,626` translation sequences).
* [X] Stripped translation instructions from prompts via case-insensitive regex.
* [X] Normalized line endings (`\r\n` ➔ `\n`) and collapsed excess whitespace (`\n{3,}` ➔ `\n\n`).
* [X] Deduplicated database entries using the code column directly (preserving `8,613` unique samples).
* [X] Formatted vocabulary and tokenized via `pygments`.
* [X] **Mutation Engine** with balanced 50/50 target distribution.
* [X] **Token-to-Index Encoding:** Convert token strings to integer sequence tensors using `vocab.json`.
* [X] **Sequence Standarization:** Truncate and pad array shapes to fixed length arrays of `512`.
* [X] **Data Serialization:** Split data using a strict 80/10/10 stratified split (by label) into training, validation, and test allocations with zero leakage.

---

## 📦 Deliverables & Handoff Interoperability

Data structures are decoupled from frameworks and written using highly stable numpy matrices rather than custom framework types.


| Deliverable Artifact   | Type         | Explicit Consumer | Internal Schema / Purpose                                                      |
| :----------------------- | :------------- | :------------------ | :------------------------------------------------------------------------------- |
| **`vocab.json`**       | JSON Dict    | ML Engineer       | `token_string ➔ integer_index` mapping for Embedding initialization.          |
| **`error_types.json`** | JSON Dict    | Both Leads        | `error_string ➔ integer_index` mapping for Multi-Class categorization.        |
| **`train.npz`**        | NumPy Binary | ML Engineer       | Contains`input_ids (N, 512)`, `labels (N,)`, and `error_types (N,)` arrays.    |
| **`val.npz`**          | NumPy Binary | Training Lead     | Identical structure; dedicated to evaluation metric sweeps and tuning.         |
| **`test.npz`**         | NumPy Binary | Training Lead     | **Held-out dataset.** Locked until final evaluation benchmark.                 |
| **`vhdl_dataset.csv`** | CSV Data     | Both Leads        | Raw source string text, mutations, and matching labels for audit verification. |

### 🛠️ Data Consumption Pattern (TensorFlow)

```python
import numpy as np
import tensorflow as tf

# Ingest underlying numpy matrices from current workspace
data = np.load("train.npz")
X_train = data["input_ids"]     # Shape: (N, 512)
y_train = data["labels"]        # Shape: (N,)
e_train = data["error_types"]   # Shape: (N,)

# Build high-throughput, prefetched pipeline 
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

```
