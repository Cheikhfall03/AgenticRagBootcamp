# NewsAI: RAG Adaptatif pour la Synthèse d'Actualités IA

Ce projet est une plateforme spécialisée dans la veille technologique sur l'intelligence artificielle. Basé sur une architecture RAG (Retrieval-Augmented Generation) avancée, il utilise LangGraph pour créer un flux de travail dynamique capable de synthétiser des actualités sur l'IA à partir de deux sources distinctes : des documents PDF fournis par l'utilisateur ou une recherche en temps réel sur le web.

![Intelligence Artificielle](https://img.shields.io/badge/AI-Intelligence%20Artificielle-blue)
![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-🦜-green)

## 🚀 Fonctionnalités Clés

- **Double Source d'Information** : Obtenez des résumés d'actualités sur l'IA soit en téléversant un article (fichier .pdf ou .txt), soit en posant directement une question pour lancer une recherche sur le web.

- **Synthèse Automatique** : Le cœur du système est sa capacité à lire le contenu (qu'il provienne d'un fichier ou du web) et à en générer un résumé concis et pertinent.

- **Recherche Web en Temps Réel** : Si vous ne fournissez pas de document, le système utilise l'API Tavily pour rechercher les dernières informations sur le sujet de l'IA qui vous intéresse.

- **Auto-Correction Intelligente** : Grâce à un mécanisme d'auto-réflexion, le système évalue la pertinence des informations trouvées et la qualité de ses propres résumés pour garantir une réponse fiable et précise.

- **Interface Utilisateur Intuitive** : L'application Streamlit permet une interaction simple : téléversez un fichier ou posez une question pour recevoir un résumé clair et direct.

## 🏛️ Architecture

L'architecture reste modulaire et robuste, mais elle est désormais optimisée pour la synthèse d'actualités.

### Interface Utilisateur (`streamlit_app.py`)
- Permet à l'utilisateur de choisir son mode d'interaction : téléverser un document ou poser une question
- Traite les fichiers téléversés ou transmet la question au moteur du graphe
- Affiche le résumé final de manière conversationnelle

### Moteur du Graphe (`graph.py`)
- Orchestre le flux de travail avec LangGraph
- La logique principale est de diriger la requête vers le traitement de document ou la recherche web

### Nœuds et Chaînes LLM
Les composants internes (récupération, évaluation, réécriture de requête) sont maintenant appliqués soit au contenu du document, soit aux résultats de la recherche web pour produire le meilleur résumé possible.

## 🤖 Modèles et Composants Techniques

Le projet s'appuie sur une sélection de modèles et de technologies de pointe pour assurer sa performance.

- **Fournisseur de LLM** : Groq
- **Modèle utilisé** : `gemma2-9b-it` - Particulièrement efficace pour les tâches de synthèse, de génération et de compréhension de texte
- **Modèle d'Embedding** : Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) - Utilisé pour vectoriser le contenu des documents téléversés
- **Base de Données Vectorielle** : ChromaDB - Stocke les vecteurs des documents PDF pour permettre au système de "lire" et de comprendre le contenu
- **API de Recherche Web** : Tavily AI - Le moteur de recherche pour trouver les actualités les plus récentes sur l'IA

## ⚙️ Flux de Travail d'une Requête

1. **Entrée** : L'utilisateur arrive sur l'application Streamlit

2. **Choix de l'Action** :
   - **Cas 1 (Fichier fourni)** : L'utilisateur téléverse un document. Le système le découpe, le vectorise et le stocke. Le flux RAG interne est ensuite utilisé pour extraire et résumer les points clés du document.
   - **Cas 2 (Question posée)** : L'utilisateur pose une question dans le champ de saisie. Le système active le nœud de recherche web (WEBSEARCH) pour collecter des articles et des informations pertinents.

3. **Génération du Résumé** (`generate`) : Que les informations proviennent du document ou du web, le LLM gemma2-9b-it est chargé de synthétiser les informations en un résumé clair et concis.

4. **Auto-Réflexion et Validation** : Le résumé est vérifié pour s'assurer qu'il est factuel (basé sur la source) et qu'il répond bien à la demande implicite de l'utilisateur.

5. **Affichage** : Le résumé final est présenté à l'utilisateur dans l'interface de chat.

## 🛠️ Installation et Utilisation

### Prérequis
- Python 3.10+
- Un gestionnaire de paquets comme pip

### 1. Cloner le Dépôt
```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>
```

### 2. Installer les Dépendances
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurer les Variables d'Environnement
Créez un fichier `.env` avec vos clés d'API :
```env
# .env
GROQ_API_KEY="gsk_..."
TAVILY_API_KEY="tvly-..."
```

### 4. Lancer l'Application
```bash
streamlit run streamlit_app.py
```

Ouvrez votre navigateur à `http://localhost:8501`. Vous pouvez maintenant téléverser un document ou poser une question pour obtenir un résumé des actualités sur l'IA.

## 📁 Structure du Projet
```
NewsAI/
├── streamlit_app.py    # Interface utilisateur Streamlit
├── graph.py           # Moteur du graphe LangGraph
├── requirements.txt   # Dépendances Python
├── .env              # Variables d'environnement (à créer)
└── README.md         # Ce fichier
```

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou à soumettre une pull request.

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

---

⭐ **N'oubliez pas de donner une étoile au projet si vous l'avez trouvé utile !**
