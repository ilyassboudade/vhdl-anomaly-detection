# 🚀 Détecteur d'Anomalies du code VHDL

![Python](https://img.shields.io/badge/Outils-NumPy-blue)
![Statut](https://img.shields.io/badge/Modèle-RNN-yellow)
![Licence](https://img.shields.io/badge/Licence-%C3%A0%20d%C3%A9finir-lightgrey)

> Un réseau de neurones récurrent (RNN) implémenté **manuellement en NumPy**, qui analyse du code source VHDL tokenisé pour détecter des erreurs syntaxiques et sémantiques avant simulation. Le projet évolue vers une cellule LSTM manuelle bidirectionnelle à tête multi-tâches.

## 📸 Démo

![Démo / Tableau de bord du projet](metadata/demo.png)

## 📊 Résultats
Ces chiffres sont ceux du run du **RNN vanille** :

| Métrique                                         | Valeur     |
| ------------------------------------------------- | ---------- |
| Exactitude globale (test)                          | **78,40 %** |
| Détection des erreurs syntaxiques                  | **~80 %**   |
| Sensibilité sur les erreurs sémantiques            | **~60 %**   |
| Taux de faux positifs résiduel (code sain)         | **~21 %**   |

## 🛠️ Pile technique & architecture

- **Langage :** Python 3.10
- **Données :** HuggingFace Datasets (`hdl2v/vhdl-dataset`)
- **Deep Learning :** NumPy pur
- **Tokenisation :** tokenizer regex spécifique VHDL (mots-clés, identifiants, opérateurs, nombres, symboles)
- **Stockage :** JSON (vocabulaire, types d'erreurs), NumPy NPZ (splits), CSV (audit)
- **Outils :** Git, Jupyter Notebook

### Architecture

RNN Vanille Séquentiel :

```
h_t    = tanh(E[x_t] · Wx + h_{t-1} · Wh)
logits = h_T · Wo + bo
```

**Hyperparamètres** :

| Paramètre    | Valeur                                                   |
| ------------ | --------------------------------------------------------- |
| `EMBED_DIM`  | 64                                                        |
| `HIDDEN_DIM` | 128                                                       |
| `N_BINARY`   | 2 (sain / anomalie)                                       |
| `N_MULTI`    | 4 (types de mutation, tête masquée sur les échantillons sains) |
| `LR`         | 3e-3 *(à ramener vers 1e-3 avec décroissance)*             |
| `BATCH_SIZE` | 64                                                        |
| `EPOCHS`     | 15                                                        |
| `CLIP_NORM`  | 5.0 (norme globale)                                       |

## 📦 Jeu de données et stratégies de mutation (`/data`)

Source : `hdl2v/vhdl-dataset` (HuggingFace), nettoyé et dédupliqué → **8 613 échantillons** (4 307 sains / 4 306 corrompus).

| Stratégie de mutation                                     | Type d'erreur          | Occurrences | Exemple                                          |
| ----------------------------------------------------------- | ------------------------ | :---------: | ------------------------------------------------- |
| Suppression d'éléments structurels essentiels                | **Structural Havoc**     |    1 079    | Incohérence entité/architecture                    |
| Désorganisation des instructions internes                    | **Scramble Architecture**|     947     | Instructions logiques mélangées dans `architecture`|
| Corruption de syntaxe (`=>` ↔ `<=`, réordonnancement)        | **Syntax Corruption**    |    1 126    | Opérateurs d'affectation inversés                  |
| Injection de code inutile                                    | **Dead Code Injection**  |    1 154    | Signal déclaré et jamais utilisé                   |
| —                                                             | **Sain (None)**          |    4 307    | —                                                   |

Séquences complétées/tronquées à **512 tokens**.

Livrables (`/data`) :

| Fichier              | Type            | Schéma                                                              |
| --------------------- | ---------------- | ---------------------------------------------------------------------- |
| `vocab.json`           | JSON             | `token_string ➔ index` (dont `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`)        |
| `error_types.json`     | JSON             | `error_string ➔ index` (les 4 catégories ci-dessus)                    |
| `train.npz`            | NumPy Binaire    | `input_ids (N, 512)`, `labels (N,)`, `error_types (N,)`                |
| `val.npz`              | NumPy Binaire    | Structure identique, dédié au réglage des hyperparamètres              |
| `test.npz`             | NumPy Binaire    | **Jeu de test verrouillé** jusqu'à l'évaluation finale                 |
| `vhdl_dataset.csv`      | CSV              | Code brut, mutations et labels correspondants, pour audit              |
