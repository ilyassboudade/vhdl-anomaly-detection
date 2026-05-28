import re, json, fsspec, random, os
# Dataset Preparation
import pandas as pd
import numpy as np
# Tokenization
from pygments.lexers import VhdlLexer
from pygments.token import Token
# Vocabulary
from collections import Counter

# Ensure the output directory exists so the script doesn't crash on save
os.makedirs("data", exist_ok=True)

# --- Step 1: VHDL Code Cleaning ---

print("Step 1: Downloading and cleaning dataset from Hugging Face...")
df = pd.read_json("hf://datasets/hdl2v/vhdl-dataset/vhdl_data.json")

df['code'] = df['prompt'].apply(
    lambda x: re.sub(r"translate the following VHDL to verilog\s*", "", str(x), flags=re.IGNORECASE)
)
df['code'] = df['code'].str.upper()
df['code'] = df['code'].str.replace('\r\n', '\n').str.replace('\r', '\n')
df['code'] = df['code'].apply(lambda x: re.sub(r'\n{3,}', '\n\n', x).strip())

df.drop_duplicates(subset=['code'], inplace=True)
df = df.drop(columns=['prompt', 'chosen'])


# --- Step 2: Tokenization Engine Core ---

lexer_vhdl = VhdlLexer()
def vhdl_tokenizer(code):
    tokens = []
    for token_type, value in lexer_vhdl.get_tokens(code):
        value = value.strip()
        if not value:
            continue
        if token_type in Token.Name or token_type in Token.Keyword:
            value = value.upper()
        tokens.append(value)
    return tokens


# --- Step 3: Mutation Engine Core ---

def introduce_semantic_error(vhdl_code):
    """
    Randomly applies one of three semantic mutations to the VHDL code.
    Returns (mutated_code, error_type_string)
    """
    strategies = ['inverted_assignment', 'relational_swap', 'logical_swap']
    random.shuffle(strategies) 
    lines = vhdl_code.split('\n')
    
    for strategy in strategies:
        # STRATEGY 1: Inverted Assignment
        if strategy == 'inverted_assignment':
            if "<=" in vhdl_code and "NOT" not in vhdl_code:
                valid_line_indices = [
                    i for i, line in enumerate(lines) 
                    if "<=" in line and ";" in line and "PORT" not in line and "GENERIC" not in line
                ]
                if valid_line_indices:
                    target_idx = random.choice(valid_line_indices)
                    parts = lines[target_idx].split("<=")
                    lines[target_idx] = f"{parts[0]}<= NOT ({parts[1].strip(' ;')});"
                    return '\n'.join(lines), "Inverted Assignment"

        # STRATEGY 2: Relational Operator Swap
        elif strategy == 'relational_swap':
            relational_map = {
                ">=": ">",
                "<=": "<",  
                ">": ">=",
                "<": "<=",
                "=": "/=",
                "/=": "="
            }
            
            valid_line_indices = []
            for i, line in enumerate(lines):
                if "PORT" in line or "GENERIC" in line:
                    continue
                if any(op in line for op in relational_map.keys()) and ("IF" in line or "WHEN" in line or "ELSIF" in line):
                    valid_line_indices.append(i)

            if valid_line_indices:
                target_idx = random.choice(valid_line_indices)
                line = lines[target_idx]
                present_ops = [op for op in relational_map.keys() if op in line]
                present_ops.sort(key=len, reverse=True)
                chosen_op = present_ops[0]
                
                lines[target_idx] = line.replace(chosen_op, relational_map[chosen_op], 1)
                return '\n'.join(lines), "Relational Swap"

        # STRATEGY 3: Logical Operator Swap
        elif strategy == 'logical_swap':
            logical_patterns = {
                r'\bAND\b': 'OR',
                r'\bOR\b': 'AND',
                r'\bXOR\b': 'XNOR',
                r'\bRISING_EDGE\b': 'FALLING_EDGE',
                r'\bFALLING_EDGE\b': 'RISING_EDGE'
            }
            
            valid_line_indices = [
                i for i, line in enumerate(lines) 
                if any(re.search(pat, line) for pat in logical_patterns.keys())
            ]
            
            if valid_line_indices:
                target_idx = random.choice(valid_line_indices)
                line = lines[target_idx]
                
                present_pats = [pat for pat in logical_patterns.keys() if re.search(pat, line)]
                chosen_pat = random.choice(present_pats)
                
                lines[target_idx] = re.sub(chosen_pat, logical_patterns[chosen_pat], line, count=1)
                return '\n'.join(lines), "Logical Swap"
                
    return vhdl_code, "None"


# --- Step 4: Strict Dataset Isolation and Balancing (1000 Total Samples) ---

print("Step 2: Isolating balancing frames...")
TOTAL_TARGET_SAMPLES = 1000
ANOMALY_RATIO = 0.5  
TARGET_ANOMALIES = int(TOTAL_TARGET_SAMPLES * ANOMALY_RATIO)  # 500
TARGET_CLEAN = TOTAL_TARGET_SAMPLES - TARGET_ANOMALIES         # 500

# 1. Sample from an oversized pool to guarantee 500 successful strategy modifications
df_mutation_pool = df.sample(n=min(1500, len(df)), random_state=42).copy()

print("Step 3: Generating mutation subsets...")
mutated_outputs = df_mutation_pool['code'].apply(introduce_semantic_error)
df_mutation_pool['code'] = [res[0] for res in mutated_outputs]
df_mutation_pool['error_type'] = [res[1] for res in mutated_outputs]

# FIXED: Replaced 'df_' with 'df_mutation_pool'
df_anomalies = df_mutation_pool[df_mutation_pool['error_type'] != "None"].copy()
df_anomalies['anomaly'] = 1

# Limit down to precisely 500 row mutations
if len(df_anomalies) > TARGET_ANOMALIES:
    df_anomalies = df_anomalies.sample(n=TARGET_ANOMALIES, random_state=42)
else:
    TARGET_ANOMALIES = len(df_anomalies)
    TARGET_CLEAN = TOTAL_TARGET_SAMPLES - TARGET_ANOMALIES

# 2. Extract 500 pristine clean codes from remaining rows to avoid index leaking
df_clean_pool = df.drop(df_anomalies.index)
df_clean = df_clean_pool.sample(n=TARGET_CLEAN, random_state=42).copy()
df_clean['anomaly'] = 0
df_clean['error_type'] = "None"

print("Step 4: Running tokenization mapping over active partitions...")
df_anomalies['tokens'] = df_anomalies['code'].apply(vhdl_tokenizer)
df_clean['tokens'] = df_clean['code'].apply(vhdl_tokenizer)

# Merge the clean and dirty pools together cleanly
df_final = pd.concat([df_clean, df_anomalies], ignore_index=True)

# Shuffle rows completely so anomalies don't bunch up in one segment
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)


# --- Step 5: Vocabulary Generation ---

print("Step 5: Compiling custom token vocabulary maps...")
counter = Counter(
    token
    for token_list in df_final['tokens']
    for token in token_list
)
SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
vocab = dict(SPECIAL)
for token, freq in counter.most_common():
    if freq >= 2:
        vocab[token] = len(vocab)
        
with open("data/vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)


# --- Step 6: Safe Serialization Formatting ---

print("Step 6: Writing clean structural frames straight to disk...")
df_csv = df_final.copy()

# Replace actual code blocks newlines with single spaces so CSV parsers load cells fast
df_csv['code'] = df_csv['code'].apply(lambda x: str(x).replace('\n', ' '))
df_csv['tokens'] = df_csv['tokens'].apply(lambda x: " ".join(x) if isinstance(x, list) else "") 

df_csv.to_csv("data/vhdl_dataset.csv", index=False)

# --- Step 7: Token-to-Index Encoding & Sequence Standardization ---
print("Step 7: Encoding tokens and standardizing sequences to length 512...")

# 1. Map your anomaly error types to integer categories for multi-class classification
# Types: 'None' -> 0, 'Inverted Assignment' -> 1, 'Relational Swap' -> 2, 'Logical Swap' -> 3
error_categories = {
    "None": 0,
    "Inverted Assignment": 1,
    "Relational Swap": 2,
    "Logical Swap": 3
}

with open("data/error_types.json", "w") as f:
    json.dump(error_categories, f, indent=2)

MAX_SEQ_LEN = 512
PAD_IDX = vocab["<PAD>"]
UNK_IDX = vocab["<UNK>"]
BOS_IDX = vocab["<BOS>"]
EOS_IDX = vocab["<EOS>"]

def encode_and_pad(token_list):
    """
    Converts a token list to a fixed-length numerical array of 512 items.
    Structures sequence as: <BOS> + tokens + <EOS> + <PAD>...
    """
    # Map tokens to vocab IDs, falling back to <UNK> if not found
    encoded = [vocab.get(token, UNK_IDX) for token in token_list]
    
    # Pack with sequence boundary markers
    full_sequence = [BOS_IDX] + encoded + [EOS_IDX]
    
    # Enforce Sequence Standardization (Length = 512)
    if len(full_sequence) >= MAX_SEQ_LEN:
        # Truncate if it exceeds max length, leaving room for the mandatory EOS marker
        return full_sequence[:MAX_SEQ_LEN-1] + [EOS_IDX]
    else:
        # Pad up to 512 using the <PAD> token ID
        padding_length = MAX_SEQ_LEN - len(full_sequence)
        return full_sequence + [PAD_IDX] * padding_length

# Vectorize the complete token corpus
numerical_sequences = np.array([encode_and_pad(t) for t in df_final['tokens']], dtype=np.int32)
binary_labels = df_final['anomaly'].to_numpy(dtype=np.int32)
multiclass_labels = df_final['error_type'].map(error_categories).to_numpy(dtype=np.int32)


# --- Step 8: Stratified Data Serialization (80/10/10 Split) ---
print("Step 8: Splitting data using a strict stratified allocation...")

def stratified_split_indices(labels, train_frac=0.8, val_frac=0.1):
    """
    Computes indices for an 80/10/10 train/val/test split 
    stratified by the multi-class error distribution using pure NumPy.
    """
    train_idx, val_idx, test_idx = [], [], []
    unique_classes = np.unique(labels)
    
    # Ensure reproducible splitting state
    rng = np.random.default_rng(42)
    
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        
        n_total = len(cls_indices)
        n_train = int(n_total * train_frac)
        n_val = int(n_total * val_frac)
        
        train_idx.extend(cls_indices[:n_train])
        val_idx.extend(cls_indices[n_train:n_train + n_val])
        test_idx.extend(cls_indices[n_train + n_val:])
        
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

# Compute allocations using multiclass targets to ensure error distribution balance
train_idxs, val_idxs, test_idxs = stratified_split_indices(multiclass_labels)

# Extract final feature and target matrices
X_train, y_train, e_train = numerical_sequences[train_idxs], binary_labels[train_idxs], multiclass_labels[train_idxs]
X_val, y_val, e_val = numerical_sequences[val_idxs], binary_labels[val_idxs], multiclass_labels[val_idxs]
X_test, y_test, e_test = numerical_sequences[test_idxs], binary_labels[test_idxs], multiclass_labels[test_idxs]


# --- Step 9: Save High-Performance NumPy Binaries ---
print("Step 9: Writing NumPy binaries out cleanly...")

np.savez_compressed("data/train.npz", input_ids=X_train, labels=y_train, error_types=e_train)
np.savez_compressed("data/val.npz", input_ids=X_val, labels=y_val, error_types=e_val)
np.savez_compressed("data/test.npz", input_ids=X_test, labels=y_test, error_types=e_test)

print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")