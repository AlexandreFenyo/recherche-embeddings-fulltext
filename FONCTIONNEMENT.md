# Fonctionnement

Cette application est un **banc d'essai de recherche documentaire** : elle indexe
des documents Markdown et permet de comparer deux familles de recherche —
**sémantique** (embeddings) et **plein texte** (mots-clés) — à granularité égale.

## Vue d'ensemble

```
Document .md
   │
   ├─ Découpage en chunks (4 stratégies au choix)
   │      │
   │      ├─ Embeddings (4 modèles) ─────────► Qdrant   (1 collection par modèle × stratégie)
   │      │
   │      └─ Texte des chunks ───────────────► OpenSearch (1 document par chunk)
   │
   └─ Recherche
          ├─ Embeddings : similarité cosinus  (+ reranking optionnel, + HyDE optionnel)
          └─ Full-text  : requête booléenne OpenSearch
```

## Composants

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Gradio + orchestration |
| `chunkers.py` | 4 stratégies de découpage |
| `embedder.py` | Modèles d'embeddings, cross-encoder de reranking, LLM HyDE |
| `indexer.py` | Indexation Qdrant + OpenSearch |
| `searcher.py` | Recherche embeddings (+ rerank, + HyDE) et full-text |
| `naming.py` | Nommage partagé des collections (source unique) |

## 1. Découpage (chunking) — `chunkers.py`

Quatre stratégies, sélectionnables à l'indexation :

- **Taille fixe** : fenêtres de N mots avec recouvrement (sliders *Taille chunk* / *Overlap*).
- **Paragraphes Markdown** : découpe sur les titres `#` et doubles sauts de ligne, fusion des fragments trop courts, re-découpe des blocs trop longs.
- **Phrases spaCy** : segmentation en phrases (modèle français `fr_core_news_md`), N phrases par chunk.
- **Chunks sémantiques** : regroupe les phrases tant que leur similarité cosinus reste élevée.

## 2. Embeddings — `embedder.py`

Quatre modèles multilingues (chargés sur GPU si disponible) :

| Modèle | Dim | Préfixe `query:`/`passage:` |
|---|---|---|
| `intfloat/multilingual-e5-large` | 1024 | oui |
| `BAAI/bge-m3` | 1024 | non |
| `intfloat/multilingual-e5-base` | 768 | oui |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | non |

Les modèles **E5** sont *asymétriques* (entraînés question → passage) et reçoivent
automatiquement le préfixe `query:` (recherche) ou `passage:` (indexation). Le
modèle **mpnet** est *symétrique* (similarité de phrases) — c'est là que HyDE
apporte le plus.

## 3. Indexation — `indexer.py`

Pour un document et une stratégie donnés :

- **Qdrant** : une **collection par couple (modèle × stratégie)**, nommée
  `{modèle}__{stratégie}`. Chaque chunk devient un point (vecteur + payload :
  `filename`, `chunk_index`, `strategy`, `model`, texte). Ré-indexer un fichier
  **remplace** ses points (purge par `filename`) — pas de doublons.
- **OpenSearch** : **un document par chunk** (champ `content` = texte du chunk,
  + `filename`, `chunk_index`, `strategy`). Même découpage que les embeddings →
  comparaison équitable. Ré-indexer purge les chunks du fichier puis réinsère.

## 4. Recherche par embeddings — `searcher.py`

- Interroge **toutes les collections** d'un coup. Résultats **regroupés par
  collection** (colonnes *Modèle* + *Stratégie*).
- **Top K** = nombre de résultats **par collection** (les scores ne sont pas
  comparables d'un modèle à l'autre, d'où le regroupement plutôt qu'un classement
  global mélangé).
- La requête n'est encodée qu'une fois par modèle (mutualisée entre collections
  d'un même modèle).

### Reranking (optionnel)
Case *Activer le reranking* : pour chaque collection, on récupère un vivier de
*candidats* (slider), puis le cross-encoder **`BAAI/bge-reranker-v2-m3`** note
chaque paire `(requête, chunk)` et on garde les Top K reclassés. La colonne
*Score CE* affiche le score du cross-encoder (logit brut : seul l'ordre compte).

### HyDE (optionnel)
Case *Activer HyDE* : le LLM **`Qwen/Qwen2.5-7B-Instruct`** génère un
**pseudo-document** (réponse hypothétique à la question). Chaque collection est
alors interrogée **deux fois** :
- variante **« requête seule »** (question brute),
- variante **« HyDE »** (pseudo-document encodé à la place de la question).

La colonne *Variante* distingue les deux, ce qui permet de **mesurer l'apport de
HyDE** modèle par modèle. Le pseudo-document généré est affiché sous la requête.
Si le reranking est aussi actif, il évalue toujours la **requête originale**.

## 5. Recherche full-text — `searcher.py`

Recherche **OpenSearch** sur les chunks, avec syntaxe booléenne :
`mot`, `mot*`, `"phrase exacte"`, `A AND B`, `A OR B`, `(A OR B) AND C`.
Retourne un extrait surligné par chunk. Comme l'indexation est au niveau chunk,
les résultats sont à la même granularité que les embeddings (colonnes *Fichier* +
*Chunk*), ce qui permet la **comparaison directe** des deux techniques.

## 6. Gestion mémoire (VRAM)

- Embeddings, reranker et LLM HyDE sont chargés **à la demande**.
- **« Charger les modèles »** charge les 4 modèles d'embeddings.
- **« Décharger les modèles »** libère **toute** la VRAM (embeddings + reranker +
  LLM) via `gc` + `torch.cuda.empty_cache()`, sans arrêter l'app.

## 7. Consoles d'inspection

- **Qdrant** : http://localhost:6333/dashboard (collections, points, vecteurs, requêtes).
- **OpenSearch Dashboards** : http://localhost:5601 (Dev Tools pour les requêtes, Discover pour parcourir l'index `documents`).

## 8. Rafraîchissement d'état

Au (re)chargement de la page (`demo.load`), l'app resynchronise l'état réel du
serveur : modèles effectivement chargés en mémoire + collections Qdrant existantes.
