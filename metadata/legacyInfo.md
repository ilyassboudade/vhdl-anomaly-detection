# Détection d'Anomalies VHDL par RNN/LSTM — Dossier Maître v3

> **Document de référence unique.** Il remplace le dossier v2 et les notes brutes issues du notebook (désormais archivées en annexe). C'est la seule source à utiliser pour rédiger le **rapport**, le **README**, l'**application**, le **site vitrine** et le **poster scientifique** — ne recopiez plus depuis le notebook directement.

**Équipe :** Ilyass BOUDADE · Zakaria ATIK · Youssef DOUIBA
**Encadrement :** Pr. Salma Hannouni · Pr. El Habib Benlahmar *(orthographe à confirmer auprès du corps enseignant — voir §0)*
**Affiliation :** Licence d'Excellence en Intelligence Artificielle (FS Ben M'Sik) — Université Hassan II de Casablanca
**Cadre :** Projet de Fin de Module (PFM) — Deep Learning

### Ce qui change depuis le v2

- La taille du corpus est tranchée et sourcée : **8 613** (§0).
- Trois nouvelles incohérences détectées dans le code du pipeline actuel (§0, lignes 6–8 du tableau) — à corriger avant de les recopier dans le rapport.
- Le pipeline actuel (nettoyage → mutation → tokenisation → encodage) est condensé en un seul endroit (§1) au lieu d'être dispersé sur huit sous-sections narratives.
- Le code et les résultats du **run baseline** (RNN vanille, avant LSTM) sont conservés en **Annexe A/B** : ils servent de ligne de base chiffrée pour le tableau d'ablation (§2.6), pas de code à garder tel quel.
- Un mini **design system commun** (§5bis) évite que l'app et le site divergent visuellement.

---

## 0. Faits de référence — ne plus les faire varier

| #   | Point                              | État constaté                                                                                                                                                                                            | Valeur / correction à utiliser partout                                                                                                                                                                                                                                                                                                                                      |
| --- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Taille du corpus                   | « 8 626 » circulait sans source ; le tableau de distribution des erreurs (§1) additionne à 8 613                                                                                                         | **8 613** échantillons (4 307 sains / 4 306 corrompus). Si 8 626 correspond à un total *avant* dédoublonnage, le documenter comme tel dans le rapport (§ Données) ; sinon, abandonner ce chiffre.                                                                                                                                                                           |
| 2   | Granularité                        | Le tokenizer est un découpeur regex spécifique VHDL (mots-clés, identifiants, opérateurs, nombres, symboles)                                                                                             | **Token-level**, jamais « caractère par caractère » dans le rapport ou le README.                                                                                                                                                                                                                                                                                           |
| 3   | Métrique de résultat               | Une seule exactitude (78,40 %) circule dans les slides/README provisoires                                                                                                                                | Toujours donner **Accuracy + F1 + ROC-AUC + matrice de confusion + rappel par type d'erreur**, jamais l'exactitude seule (voir §2.6 et §3).                                                                                                                                                                                                                                 |
| 4   | Split train/val/test               | Non documenté dans le pipeline actuel                                                                                                                                                                    | Documenter explicitement le split **par `source_file_id`, avant mutation** (voir §2.1 bug #1), avec les proportions exactes et la graine — à extraire du notebook et à reporter ici une fois pour toutes.                                                                                                                                                                   |
| 5   | Lecture des résultats manuels      | « code sain identifié comme sain à 21 % » est ambigu                                                                                                                                                     | Reformuler : **taux de faux positifs résiduel ≈ 21 %** sur le code sain, baseline RNN vanille (voir Annexe B).                                                                                                                                                                                                                                                              |
| 6   | Étiquettes de la tête multi-classe | Le code fixe `N_MULTI = 4  # None / Inverted / Relational / Logical`, mais les catégories réelles du dataset sont *Dead Code Injection, Structural Havoc, Scramble Architecture, Syntax Corruption* (§1) | Le commentaire de code date d'un schéma d'étiquetage périmé. `N_MULTI = 4` reste correct **si** la tête multi-classe est bien masquée sur les échantillons sains (pas de classe « None » à prédire, cf. Axe 1 §2.4) — mais renommer les 4 classes dans `error_types.json` et dans le code pour qu'elles correspondent aux vraies catégories, avant toute figure du rapport. |
| 7   | Backpropagation du baseline        | Le `train_step` actuel (Annexe A) ne montre de gradient calculé que pour `Wo`/`bo` (`dWo = np.zeros_like(self.Wo)` puis mise à jour)                                                                     | À vérifier dans le notebook complet : si `Wx`, `Wh` et la matrice d'embedding `E` ne sont réellement jamais mis à jour, la stagnation à ~78 % s'explique autant par ce bug que par l'évanouissement du gradient. À confirmer avant de l'attribuer uniquement au RNN vanille dans le rapport.                                                                                |
| 8   | Orthographe encadrant              | « BEN LAHMAR » (dossier v2) vs « Benlahmar » (rapport)                                                                                                                                                   | Confirmer l'orthographe officielle avant l'impression finale du rapport et de la page de garde.                                                                                                                                                                                                                                                                             |

---

## 1. Pipeline actuel — résumé factuel (ne pas refaire, juste documenter)

Ces étapes existent déjà dans le notebook et fonctionnent ; elles sont condensées ici pour alimenter directement le rapport (§3 Données) et le README (§8) sans avoir à rouvrir le notebook. Le code correspondant est en **Annexe A**.

**Source et nettoyage**

- Source : `hdl2v/vhdl-dataset` sur Hugging Face, colonnes `prompt` / `chosen`.
- Nettoyage : suppression du préfixe d'instruction (« translate the following VHDL to verilog »), passage en majuscules, normalisation des retours à la ligne, suppression des doublons sur `code`, suppression des colonnes `prompt`/`chosen`.

**Génération des anomalies** (`introduce_semantic_error`, détection des blocs `BEGIN`/`END`)
| Stratégie | Type d'erreur | Occurrences | Exemple |
|---|---|---|---|
| Suppression d'éléments structurels essentiels | **Structural Havoc** | 1 079 | Incohérence entité/architecture (ex. `architecture wrong_arch of adder`) |
| Désorganisation des instructions internes | **Scramble Architecture** | 947 | Instructions logiques mélangées dans le bloc `architecture` |
| Corruption de syntaxe (`=>` ↔ `<=`, réordonnancement) | **Syntax Corruption** | 1 126 | Opérateurs d'affectation inversés |
| Injection de code inutile | **Dead Code Injection** | 1 154 | Signal déclaré et jamais utilisé |
| — | **None (sain)** | 4 307 | — |
| | **Total** | **8 613** | 4 306 corrompus / 4 307 sains |

- Le pool muté est échantillonné (`random_state=42`) et les indices sources utilisés pour générer une anomalie sont retirés du pool sain, pour éviter la fuite classe saine ↔ classe corrompue **au niveau de l'échantillon**. Cela ne remplace pas le split par fichier exigé en §2.1 : c'est une garantie différente (pas de doublon de contenu entre classes), pas une garantie contre la fuite train/val/test.

**Tokenisation, vocabulaire, encodage**

- Tokenizer regex spécifique VHDL → tokens normalisés en majuscules.
- Vocabulaire construit par fréquence, tokens spéciaux `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`, sauvegardé en JSON.
- Séquences encodées, complétées/tronquées à **512 tokens**, exportées en `train.npz` / `val.npz` / `test.npz` avec label binaire (sain/corrompu) et label multiclasse (type d'erreur, masqué sur les échantillons sains).

---

## 2. Axe 1 — Implémentation manuelle du réseau (priorité n°1)

C'est la valeur académique du projet. Le « from scratch NumPy » doit être irréprochable, pas seulement fonctionnel.

### 2.1 Trois bugs silencieux à vérifier en premier

1. **Fuite de données.** Si un même fichier VHDL source produit une version saine *et* sa version mutée, et que ces deux échantillons tombent de part et d'autre du split, le modèle mémorise le fichier au lieu d'apprendre l'anomalie. **Splitter par `source_file_id` avant mutation**, jamais après. C'est la première question d'un jury.
2. **Masque de padding absent.** Avec `<PAD>=0` et une longueur fixe de 512, dérouler le RNN sur le padding écrase l'état caché utile. Il faut propager un masque `mask[t] ∈ {0,1}` : `h_t = mask_t ⊙ h̃_t + (1 - mask_t) ⊙ h_{t-1}`.
3. **Raccourcis d'apprentissage.** Injecter des mots-clés Verilog (`wire`, `reg`) ou supprimer `LIBRARY` crée un détecteur de mot-clé, pas un détecteur d'anomalie. Mesurer la performance en retirant cette mutation : si l'exactitude s'effondre, le modèle trichait.

*(Voir aussi §0.7 : vérifier que le RNN actuel rétropropage bien sur `Wx`/`Wh`/`E`, pas seulement sur la couche de sortie, avant de conclure que le LSTM est seul responsable des gains.)*

### 2.2 Passage d'un RNN vanilla à une cellule LSTM manuelle

La saturation à ~60 % sur les erreurs sémantiques vient du gradient qui s'évanouit sur 512 pas. Implémenter la cellule LSTM en NumPy, portes concaténées en une seule matrice pour un seul produit matriciel par pas de temps :

```
z_t = W_x · x_t + W_h · h_{t-1} + b        # (4H,)
i_t = σ(z_t[0:H])        # porte d'entrée
f_t = σ(z_t[H:2H])       # porte d'oubli
o_t = σ(z_t[2H:3H])      # porte de sortie
g_t = tanh(z_t[3H:4H])   # candidat
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

Rétropropagation à implémenter à la main (BPTT), en accumulant `dW_x`, `dW_h`, `db` sur tous les pas de temps.

**Détails qui font la différence :**

- **Biais d'oubli initialisé à +1.0** : la porte d'oubli s'ouvre par défaut, le gradient circule dès l'époque 1.
- **Initialisation** : Xavier/Glorot pour `W_x`, orthogonale pour `W_h` (essentiel en récurrent).
- **Vérification analytique du gradient** (*gradient checking*) contre des différences finies, sur un mini-modèle. À montrer dans le rapport : c'est la preuve que l'implémentation manuelle est correcte.
- **Clipping par norme globale** (et non par élément) au seuil 5.0.

### 2.3 Agrégation de séquence

Ne pas classifier sur `h_T` seul : avec du padding et 512 pas, cet état est appauvri. Utiliser un **mean-pooling masqué** sur les états cachés, ou concaténer `[mean-pool ; max-pool]`. Gain typique immédiat de plusieurs points, pour ~10 lignes de code.

### 2.4 Bi-directionnalité et tête multi-tâches

- **Bi-LSTM** : une passe avant + une passe arrière, concaténation → `2H`. Une erreur de délimiteur se détecte souvent mieux par le contexte droit.
- **Deux têtes sur le même encodeur** : sortie sigmoïde (sain/corrompu) + sortie softmax (type d'erreur parmi les 4 catégories de mutation, `error_types.json` — voir §0.6 pour la correction des noms de classes). Perte combinée `L = L_bce + λ · L_ce`, λ ≈ 0.5, la tête multi-classe **masquée sur les échantillons sains** (pas de classe « None » à prédire).

### 2.5 Entraînement

- Adam avec **correction de biais** explicite (souvent oubliée en implémentation manuelle) ; `lr = 1e-3` avec décroissance — `3e-3` (valeur actuelle, voir Annexe A) est agressif pour un récurrent.
- **Dropout** sur l'embedding et avant la couche de sortie (masque inversé, actif en entraînement uniquement).
- **Early stopping** sur la F1 de validation, pas sur la perte.
- Journaliser chaque époque en JSON (perte train/val, métriques, norme du gradient) : ce fichier alimente directement les courbes du rapport et du site.

### 2.6 Ablation à produire

Un tableau d'ablation vaut plus qu'un chiffre isolé. La ligne « baseline » est déjà chiffrée (Annexe B) ; complétez les suivantes au fur et à mesure.

| Configuration                                   | Accuracy | F1  | Rappel sémantique |
| ----------------------------------------------- | -------- | --- | ----------------- |
| RNN vanilla + dernier état (baseline, Annexe B) | 78,4 %   | —   | ~60 %             |
| + masque de padding                             |          |     |                   |
| + mean-pooling masqué                           |          |     |                   |
| LSTM manuel                                     |          |     |                   |
| Bi-LSTM + tête multi-tâches                     |          |     |                   |

---

## 3. Axe 2 — Rapport

Structure cible, 12–15 pages, orientée démonstration scientifique :

1. **Introduction** — limites des linters/compilateurs, positionnement du problème.
2. **État de l'art** — vérification HDL, RNN/LSTM, détection d'anomalies sur code.
3. **Données** — corpus source (§1, 8 613 échantillons sourcés), pipeline de mutation, **split par fichier source** (§2.1), statistiques et exemples avant/après mutation.
4. **Méthode** — équations LSTM (§2.2), dérivation du BPTT, preuve par gradient checking, schéma d'architecture.
5. **Protocole expérimental** — hyperparamètres, matériel, temps d'entraînement, graine aléatoire.
6. **Résultats** — courbes, matrice de confusion, métriques par type d'erreur, tableau d'ablation (§2.6).
7. **Discussion** — pourquoi la sémantique résiste, analyse de cas d'échec commentés (2 ou 3 extraits de code, cf. Annexe B pour le test manuel AND).
8. **Limites et perspectives** — troncature à 512 tokens, mutations synthétiques vs erreurs réelles, Transformers.
9. **Conclusion, références, annexes** (tableau reproductibilité).

**Règle de fond :** toute affirmation de performance renvoie à une figure ou un tableau numéroté — plus de chiffre « nu » comme dans la version précédente du rapport (Annexe B).

---

## 4. Axe 3 — README

Ordre imposé par la lecture réelle d'un dépôt :

1. Titre + une phrase + badges (licence, Python, statut).
2. **GIF de démonstration** (8 s, l'app qui détecte une erreur) — l'élément le plus rentable du README.
3. Résultats en 3 chiffres, dès le haut de page (Accuracy + F1 + ROC-AUC, pas l'exactitude seule — §0.3).
4. **Quickstart en 4 commandes** : clone → `pip install -r requirements.txt` → `python predict.py --file exemple.vhd` → sortie attendue affichée.
5. Arborescence du dépôt commentée.
6. Reproduire l'entraînement (une commande, graine fixée).
7. Architecture (schéma + hyperparamètres).
8. Jeu de données et stratégies de mutation (le tableau de §1 y va directement).
9. Limites, feuille de route, citation, équipe, licence.

Un `requirements.txt` avec versions épinglées et une graine fixée sont non négociables.

---

## 5. Axe 4 — Interface de l'application

**Proposition de valeur :** coller du VHDL, obtenir un verdict et une localisation en moins d'une seconde.

**Écran unique, deux volets :**

- **Gauche — éditeur** : coloration syntaxique VHDL, 3 exemples pré-chargés (sain / erreur syntaxique / erreur sémantique), import de fichier `.vhd`, compteur de tokens avec avertissement au-delà de 512.
- **Droite — verdict** : badge SAIN/ANOMALIE, jauge de confiance, type d'erreur prédit avec probabilités (4 catégories, §0.6), et surtout la **carte de saillance** — intensité de la contribution par token, reprojetée sur le code. Sans elle, l'outil est une boîte noire ; avec elle, il est utilisable.

**À soigner :** état de chargement, cas « code vide » et « trop long », affichage honnête de l'incertitude quand la confiance est faible (< 0,65 : « incertain »), et un panneau repliable « comment ça marche » qui affiche les états cachés.

## 5bis. Design system commun (app + site)

Pour que l'app et le site vitrine ne divergent pas visuellement (ils partagent la même démo) :

- **Sombre par défaut**, une seule couleur d'accent en dehors du verdict.
- **Monospace** pour tout ce qui est code ou token.
- **Rouge/vert réservés exclusivement au verdict** (sain/anomalie) — jamais utilisés ailleurs dans l'UI (boutons, liens), pour que le verdict reste immédiatement reconnaissable.
- La carte de saillance de l'app doit pouvoir être réutilisée telle quelle dans la section « Démo interactive » du site (§6.3) plutôt que redéveloppée.

---

## 6. Axe 5 — Site vitrine

Une page, scroll vertical, six sections :

1. **Hero** — titre, sous-titre d'une ligne, deux boutons (Essayer la démo / GitHub), fond animé discret.
2. **Problème** — trois cartes : les linters s'arrêtent à la grammaire, la simulation coûte cher, les erreurs logiques passent.
3. **Démo interactive intégrée** — la même que l'app (§5bis), au-dessus de la ligne de flottaison si possible.
4. **Comment ça marche** — schéma du pipeline en 4 étapes (§1), animé au scroll.
5. **Résultats** — chiffres clés, courbe de convergence, tableau d'ablation (§2.6).
6. **Équipe, encadrement, affiliation, remerciements, liens** (rapport PDF, poster, vidéo, dépôt).

Argument différenciant à afficher en clair : **réseau implémenté intégralement en NumPy, sans framework de deep learning.** C'est ce qui distingue le projet.

---

## 7. Axe 6 — Poster scientifique (à refaire)

> **Note :** le poster existant est à refaire avec un outil basé texte/versionné plutôt qu'un éditeur graphique — trois options courantes, à choisir selon la contrainte principale :

| Outil                                                         | Quand le choisir                                                                                                                 | Point d'attention                                                                                                                   |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Marp** (Markdown + CSS)                                     | Le plus rapide à démarrer ; on connaît déjà Markdown (comme ce dossier) ; export direct en PDF/PNG.                              | Mise en page moins fine que LaTeX pour un grand format A0 ; soigner le thème CSS pour éviter l'effet « slide agrandie ».            |
| **Beamer/LaTeX (posterbeamer, `tikzposter`, `beamerposter`)** | Le rapport est déjà en LaTeX, ou une mise en page académique très maîtrisée (colonnes, équations LSTM de §2.2) est prioritaire.  | Compilation plus lente, syntaxe plus lourde pour itérer vite.                                                                       |
| **Typst**                                                     | On veut la rigueur de LaTeX (équations, mise en page précise) avec une syntaxe plus légère et une compilation quasi instantanée. | Écosystème de templates poster plus jeune ; vérifier qu'un template A0 correspond au format demandé par l'école avant de s'engager. |

**Contenu à reprendre directement de ce dossier** (pas de nouvelle rédaction) :

- Titre, équipe, encadrement, affiliation (en-tête de ce document).
- Problème + argument différenciant « NumPy, sans framework » (§6).
- Schéma pipeline en 4 étapes (§1) et équations LSTM (§2.2).
- Chiffres clés et tableau d'ablation (§2.6) — jamais l'exactitude seule (§0.3).
- QR code vers la démo de l'app (§5) et vers le dépôt GitHub.

---

## 8. Ordre d'exécution conseillé

| #   | Tâche                                                                                                   | Effort       | Impact             |
| --- | ------------------------------------------------------------------------------------------------------- | ------------ | ------------------ |
| 1   | Vérifier la fuite de données (split par fichier) **+ confirmer le backprop complet du baseline (§0.7)** | Faible       | Critique           |
| 2   | Masque de padding + mean-pooling                                                                        | Faible       | Élevé              |
| 3   | Cellule LSTM manuelle + gradient checking                                                               | Élevé        | Élevé              |
| 4   | Métriques complètes + ablation                                                                          | Moyen        | Élevé              |
| 5   | Interface avec carte de saillance                                                                       | Moyen        | Élevé (démo)       |
| 6   | README + GIF                                                                                            | Faible       | Élevé (visibilité) |
| 7   | Rapport restructuré                                                                                     | Élevé        | Élevé (note)       |
| 8   | Site vitrine                                                                                            | Moyen        | Moyen              |
| 9   | Poster scientifique (Marp/Beamer/Typst)                                                                 | Faible–Moyen | Moyen (soutenance) |

Les points 1 et 2 peuvent faire bouger les résultats à eux seuls : les traiter avant de relancer un entraînement long.

---

## Annexe A — Code du pipeline et du baseline actuels (référence, à ne pas conserver tel quel)

Ce code documente ce qui existe *aujourd'hui*. Il alimente §1 et sert de point de comparaison pour §2 — il n'est pas la cible finale.

**Fonctions de perte et d'exactitude**

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bce_loss_with_logits(logits, y, pos_w=1.0):
    probs = sigmoid(logits)
    loss = -(pos_w * y * np.log(probs + 1e-9) +
             (1 - y) * np.log(1 - probs + 1e-9))
    return loss.mean()

def accuracy_from_logits(logits, y):
    preds = (sigmoid(logits) > 0.5).astype(int)
    return (preds == y).mean()
```

**RNN vanille actuelle (à remplacer par la cellule LSTM de §2.2)**

```python
class RNN:
    def __init__(self, vocab_size, emb, hid):
        self.E  = np.random.randn(vocab_size, emb) * 0.01
        self.Wx = np.random.randn(emb, hid) * 0.1
        self.Wh = np.random.randn(hid, hid) * 0.1
        self.Wo = np.random.randn(hid, 1) * 0.1
        self.bo = np.zeros(1)

    def forward(self, x):
        B, T = x.shape
        h = np.zeros((B, self.Wh.shape[0]))
        for t in range(T):
            emb = self.E[x[:, t]]
            h = np.tanh(emb @ self.Wx + h @ self.Wh)
        return (h @ self.Wo + self.bo).squeeze()

    def predict(self, x):
        return 1 / (1 + np.exp(-self.forward(x)))

    def train_step(self, x, y, lr=1e-3):
        logits = self.forward(x)
        probs = 1 / (1 + np.exp(-logits))
        grad = (probs - y) / len(y)

        dWo = np.zeros_like(self.Wo)   # ⚠ voir §0.7 : jamais rempli avant la mise à jour
        dbo = grad.sum()

        self.Wo -= lr * dWo
        self.bo -= lr * dbo

        loss = -np.mean(y * np.log(probs + 1e-9) + (1 - y) * np.log(1 - probs + 1e-9))
        acc = ((probs > 0.5) == y).mean()
        return loss, acc
```

**Hyperparamètres actuels**

```python
VOCAB_SIZE = len(vocab)
EMBED_DIM  = 64     # embedding dimension
HIDDEN_DIM = 128    # RNN hidden state size
N_BINARY   = 2       # sain vs anomalie
N_MULTI    = 4       # 4 catégories de mutation, tête masquée sur les échantillons sains — voir §0.6 pour renommer le commentaire
LR         = 3e-3    # à ramener vers 1e-3 avec décroissance, cf. §2.5
BATCH_SIZE = 64
EPOCHS     = 15
CLIP_NORM  = 5.0     # norme globale, cf. §2.2
```

---

## Annexe B — Résultats bruts du run baseline (ligne de référence pour §2.6)

| Époque | Loss   | Train Acc | Val Acc |
| ------ | ------ | --------- | ------- |
| 1      | 0.9135 | 0.3813    | 0.5070  |
| 2      | 0.8063 | 0.4233    | 0.5443  |
| 3      | 0.7203 | 0.4607    | 0.5734  |
| 4      | 0.6033 | 0.5216    | 0.6321  |
| 5      | 0.4937 | 0.5908    | 0.6917  |
| 6      | 0.4700 | 0.6899    | 0.7274  |
| 7      | 0.4500 | 0.7209    | 0.7452  |
| 8      | 0.4160 | 0.7377    | 0.7658  |
| 9      | 0.4181 | 0.7672    | 0.7771  |
| 10     | 0.3948 | 0.7647    | 0.7737  |
| 11     | 0.3700 | 0.7633    | 0.7854  |
| 12     | 0.3636 | 0.7736    | 0.7853  |
| 13     | 0.3590 | 0.7643    | 0.7903  |
| 14     | 0.3499 | 0.7853    | 0.7860  |
| 15     | 0.3481 | 0.7775    | 0.7832  |

**Test final : accuracy = 0.784002306805075**

**Test manuel — porte logique AND (3 cas, à réutiliser en §3.7 du rapport)**

```
TEST: NORMAL CODE       → Anomaly probability: 0.2146 → Prediction: NORMAL
TEST: SYNTAX ERROR      → Anomaly probability: 0.7955 → Prediction: ANOMALY
TEST: SEMANTIC ERROR    → Anomaly probability: 0.5453 → Prediction: ANOMALY
```

**Lecture (reformulée conformément à §0.5) :**

- Code sain : classé correctement, avec un **taux de faux positifs résiduel de ~21 %** — le modèle vérifie surtout la syntaxe, sa réserve porte sur la partie sémantique.
- Erreur syntaxique : forte probabilité d'anomalie (0.7955) — bien détectée.
- Erreur sémantique : détectée (0.5453 > 0.5) mais avec une marge de confiance nettement plus faible que pour l'erreur syntaxique — c'est exactement le point que le LSTM + tête multi-tâches (§2.2, §2.4) doit corriger.

**Conclusion du run baseline (résumée) :** ~78 % d'exactitude globale, ~80 % de détection sur les erreurs syntaxiques, ~60 % de sensibilité sur les erreurs sémantiques, ~21 % de faux positifs résiduels sur le code sain. C'est le chiffre à battre pour chaque ligne du tableau d'ablation (§2.6).
