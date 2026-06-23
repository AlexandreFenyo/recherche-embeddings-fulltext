# Installation (WSL + Docker)

Guide d'installation de l'application de **recherche dans des documents Markdown**
(embeddings + full-text) sous **WSL2** avec **Docker**.

## 1. Prérequis

| Élément | Détail |
|---|---|
| **WSL2** | Ubuntu 22.04+ recommandé |
| **Docker** | Docker Desktop (intégration WSL) ou Docker Engine installé dans WSL |
| **Python** | 3.11+ (testé en 3.13) |
| **GPU NVIDIA** | *optionnel mais recommandé* — pilotes NVIDIA pour WSL + CUDA. Sans GPU, l'app bascule automatiquement sur CPU (plus lent). |
| **RAM / VRAM** | embeddings ≈ 7 Go VRAM, reranking +2,3 Go, LLM HyDE +15 Go. Sans GPU : prévoir de la RAM. |
| **Espace disque** | ~25 Go pour les modèles téléchargés depuis Hugging Face (au 1ᵉʳ usage). |

Vérifier Docker :
```bash
docker --version
docker compose version
```

## 2. Récupérer le code

```bash
git clone https://github.com/<utilisateur>/<repo>.git
cd <repo>
```

## 3. Démarrer l'infrastructure (Docker)

Les bases de données tournent en conteneurs :

```bash
docker compose up -d
```

Cela lance :

| Service | Port | Rôle |
|---|---|---|
| **Qdrant** | 6333 (REST), 6334 (gRPC) | base vectorielle (embeddings) — console : http://localhost:6333/dashboard |
| **OpenSearch** | 9200 | moteur full-text |
| **OpenSearch Dashboards** | 5601 | console OpenSearch : http://localhost:5601 |

Vérifier :
```bash
curl -s http://localhost:6333/healthz      # Qdrant
curl -s http://localhost:9200/_cluster/health   # OpenSearch
```

## 4. Environnement Python

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Modèle spaCy français (utilisé par 2 stratégies de découpage)
python -m spacy download fr_core_news_md
```

## 5. Lancer l'application

```bash
./run.sh
# ou : python app.py
```

L'interface est sur **http://localhost:7860**.

> `run.sh` appelle `python3.13`. Si ta version diffère, lance directement
> `python app.py` depuis l'environnement virtuel.

## 6. Premier usage

1. Cliquer **« Charger les modèles »** → télécharge (1ʳᵉ fois) et charge les 4 modèles d'embeddings depuis Hugging Face.
2. **Indexer** un document `.md` : choisir une stratégie de découpage, téléverser le fichier, cliquer « Indexer ». Cela peuple Qdrant (un vecteur par chunk × modèle) **et** OpenSearch (un document par chunk).
3. **Rechercher** : onglet *Embeddings* (sémantique) ou *Full-text* (mots-clés).

### Options avancées (onglet Embeddings)
- **Reranking** : coche la case → un cross-encoder (`BAAI/bge-reranker-v2-m3`) reclasse les candidats. Téléchargé au 1ᵉʳ usage (~1,1 Go).
- **HyDE** : coche la case → un LLM (`Qwen/Qwen2.5-7B-Instruct`) génère un pseudo-document ; la recherche est faite **avec et sans** HyDE pour comparer. Téléchargé au 1ᵉʳ usage (~15 Go).

### Libérer la VRAM
Le bouton **« Décharger les modèles »** libère la VRAM (embeddings + reranker + LLM HyDE) sans arrêter l'application.

## 7. Arrêt

```bash
# Arrêter l'app : Ctrl-C dans le terminal de run.sh
# Arrêter l'infra Docker :
docker compose down          # conserve les données (volumes)
docker compose down -v       # supprime aussi les données indexées
```

## Dépannage

- **Qdrant injoignable au démarrage de l'app** : vérifier que `docker compose up -d` a bien démarré le conteneur `qdrant` (port 6333).
- **OpenSearch « non disponible »** dans l'UI : le conteneur OpenSearch met ~30 s à démarrer ; réessayer.
- **CUDA non détecté** : l'app fonctionne en CPU (colonne *Device* = `cpu`). Pour le GPU sous WSL, installer les pilotes NVIDIA pour WSL et le runtime conteneur si besoin.
- **Mémoire GPU saturée** : utiliser « Décharger les modèles », ou ne pas activer HyDE + reranking simultanément si la VRAM est limitée.
