# Taiwanese Bankruptcy Prediction

Projet de Machine Learning (ESILV A4) : prédire la faillite d'entreprises taïwanaises à partir de 95 ratios financiers.

## Données

- Source : [Taiwanese Bankruptcy Prediction — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/572/taiwanese+bankruptcy+prediction) (Liang et al., 2016, *European Journal of Operational Research*)
- Fichier : `data.csv` — 6819 entreprises, 96 colonnes
- Cible : `Bankrupt?` (1 = faillite), classes très déséquilibrées (**3,2 %** de faillites)

## Structure du dépôt

| Fichier | Contenu |
|---|---|
| `data.csv` | Jeu de données brut |
| `DataExploration.ipynb` | Analyse exploratoire |
| `Modelisation.ipynb` | Préparation des données, entraînement et évaluation des modèles |
| `Exploration.md` | Notes de suivi : faits établis, explorations restantes, feuille de route de la modélisation, astuces |
| `requirements.txt` | Dépendances Python |

## Principe de séparation des notebooks

| Notebook | Rôle | Règle |
|---|---|---|
| `DataExploration.ipynb` | **Exposer des faits** sur les données | Aucune transformation de la donnée destinée au modèle : on observe, on mesure, on conclut en markdown |
| `Modelisation.ipynb` | **Appliquer les décisions** tirées de l'exploration, puis entraîner et évaluer les modèles | Toute transformation se fait ici, dans l'ordre de la feuille de route (section 4 de `Exploration.md`) |

> Seule exception tolérée côté exploration : `data.columns = data.columns.str.strip()`. Elle ne change aucune valeur, elle rend seulement les colonnes accessibles par leur nom. Elle devra être **refaite** dans le notebook de modélisation.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Ouvrir ensuite les notebooks dans Jupyter ou VS Code en sélectionnant le noyau `.venv`.

## Démarche

1. **Exploration** (`DataExploration.ipynb`) : qualité des données, échelles, normalisation min-max présumée, redondance entre variables (corrélation de Spearman), lien avec la cible (information mutuelle).
2. **Modélisation** (`Modelisation.ipynb`) : nettoyage, split stratifié, traitement des valeurs extrêmes, sélection gloutonne des variables, rééquilibrage des classes, puis régression logistique → Random Forest → gradient boosting.
3. **Évaluation** : rappel, F1, PR-AUC et matrice de confusion (pas d'accuracy, trompeuse avec 96,8 % de non-faillites), avec ajustement du seuil de décision.
