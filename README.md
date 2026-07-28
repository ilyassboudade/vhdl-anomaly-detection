# Détection d'Anomalies du code VHDL

> Une implémentation manuelle d'un RNN simple qui analyse le code source VHDL tokenisé pour détecter ses erreurs syntaxiques et sémantiques avant la simulation.

# Projet de Fin de Module

**Membres du groupe :**

- BOUDADE Ilyass

- ATIK Zakaria

- DOUIBA Youssef

**Encadrant :** Pr. HANNOUNI Salma

## Contenus de ce répertoire

- **Rapport succinct** détaillant le contexte, l'approche méthodologique et l'analyse des performances.

- **Poster Scientifique** résumant les détails du projet entammés dans le rapport du projet.

- **Vidéo de démonstration** présentant le comportement du modèle et les résultats obtenus.

- **Notebook Jupyter complet** contenant le code source structuré, nettoyé, documenté et exécutable.

## Description du Projet

Le projet implémente un réseau **RNN** conçu pour analyser la syntaxe séquentielle du code VHDL au niveau des caractères. Comme les erreurs logiques s'étendent souvent sur plusieurs lignes et dépendent fortement d'un contexte à longue portée, le système utilise une architecture récurrente pour mémoriser et évaluer l'état de la description matérielle.

Pour pallier la forte redondance du code VHDL, nous avons mis en place une stratégie de mutation avancée permettant d'isoler distinctement les profils de code sains des profils corrompus. Les résultats d'entraînement montrent une convergence claire de la fonction de perte et une bonne séparation des anomalies.

## 📊 Jeu de Données & Stratégie de Mutation

- **Corpus de base :** `hdl2v/vhdl-dataset` sur HuggingFace (`8626` paires d'échantillons bruts).

- **Le Défi :** Le jeu de données source est conçu pour la traduction de VHDL en Verilog. Il ne contient aucune erreur pré-étiquetée.

- **La Solution :** Pour entraîner un classificateur supervisé binaire et multi-classe, l'injection d'anomalies synthétiques est nécessaire :
  
  1. Analyser le code VHDL propre et compilable (`label = 0`, `error_type = "NONE"`).
  
  2. Introduire **exactement une stratégie d'injection d'erreur**.
  
  3. Étiqueter le code muté (`label = 1`) et le classer dans l'un des types d'erreurs désignés.

## Spécifications de Tokenisation & de Normalisation

Pour garantir une augmentation minimale du vocabulaire et maximiser la force sémantique des plongements :

- **Analyseur Lexical (Lexer) :** `pygments.lexers.VhdlLexer`

- **Insensibilité à la Casse :** Tous les jetons sont strictement convertis en majuscules via `str.upper()` pour correspondre aux spécifications du VHDL.

- **Jetons Spéciaux :**
  
  - `<PAD>` (`0`) : Élément de remplissage de séquence (padding).
  
  - `<UNK>` (`1`) : Termes hors vocabulaire interceptés lors de l'inférence.
  
  - `<BOS>` (`2`) : Indicateur de début de séquence.
  
  - `<EOS>` (`3`) : Indicateur de fin de séquence.

## 📦 Données livrables (trouvées dans le dossier "/data")

| **Artefact Livrable**  | **Type**          | **Schéma Interne / Objectif**                                                                       |
| ---------------------- | ----------------- | --------------------------------------------------------------------------------------------------- |
| **`vocab.json`**       | Dictionnaire JSON | Correspondance `token_string ➔ integer_index` pour l'initialisation du plongement.                  |
| **`error_types.json`** | Dictionnaire JSON | Correspondance `error_string ➔ integer_index` pour la catégorisation multi-classe.                  |
| **`train.npz`**        | Binaire NumPy     | Contient les tableaux `input_ids (N, 512)`, `labels (N,)` et `error_types (N,)`.                    |
| **`val.npz`**          | Binaire NumPy     | Structure identique ; dédié aux balayages des métriques d'évaluation et aux ajustements.            |
| **`test.npz`**         | Binaire NumPy     | **Jeu de données de test (Held-out).** Verrouillé jusqu'à l'évaluation finale.                      |
| **`vhdl_dataset.csv`** | Données CSV       | Texte brut des chaînes sources, mutations et étiquettes correspondantes pour vérification et audit. |
