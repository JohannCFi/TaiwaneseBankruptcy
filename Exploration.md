# Suite de l'exploration — Taiwanese Bankruptcy

> Le principe de séparation entre `DataExploration.ipynb` et `Modelisation.ipynb` est décrit dans le [README](README.md).

---

## 1. Faits déjà établis et conséquences pour la modélisation

| # | Fait observé | Comment on l'a vu | À faire dans le notebook de modélisation |
|---|---|---|---|
| F1 | Tous les noms de colonnes sauf `Bankrupt?` commencent par un espace | `data.info()`, décalage visuel, `KeyError` sur `data["Equity to Liability"]` | `data.columns = data.columns.str.strip()` juste après le `read_csv` |
| F2 | Aucune valeur manquante | `data.isna().values.any()` → `False` | Pas d'imputation nécessaire, sauf si on remplace des valeurs aberrantes par NaN (voir F5) |
| F3 | 6819 lignes, 96 colonnes, presque toutes en `float64` | `data.info()` | — |
| F4 | `Net Income Flag` vaut toujours 1 (écart-type nul) | `describe()`, comptage des valeurs ≠ 1 | **Supprimer** la colonne : elle n'apporte aucune information |
| F5 | 24 colonnes ont un max autour de 10⁹ à 10¹⁰, alors que les 72 autres sont dans [0, 1] | `describe().loc["max"] != 1` | Traitement à décider **après** l'analyse E1 (section 2) |
| F6 | Les variables semblent **normalisées min-max** sur [0, 1] (non documenté sur UCI) | Min = 0 et max = 1 sur 72 colonnes ; `Equity to Liability` et `Liability to Equity` sont toutes deux < 1, ce qui est impossible pour de vrais inverses ; `Operating Profit Rate` ≈ 0,999 partout | Les valeurs ne s'interprètent pas comme de vrais ratios, mais **l'ordre entre entreprises est conservé**. Ne pas re-normaliser les 72 colonnes déjà dans [0, 1] sans raison ; un `StandardScaler` reste utile pour la régression logistique |
| F7 | Classes très déséquilibrées : **3,2 %** de faillites (`Bankrupt? = 1`) | `data["Bankrupt?"].mean()` | Voir section 4, étapes 3, 6 et 7 : split stratifié, rééquilibrage, métriques adaptées |
| F8 | La corrélation de **Pearson** rate les redondances non linéaires : −0,16 entre `Equity to Liability` et `Liability to Equity`, contre **−0,993** en **Spearman** | `np.corrcoef` vs `.corr(method="spearman")` | Toujours utiliser **Spearman** pour repérer les variables redondantes dans ce dataset |
| F9 | Il existe de nombreuses paires très redondantes (\|Spearman\| > 0,9) : 86 paires distinctes, dont 73 positives et 13 négatives | `triu` + `stack` sur la matrice de Spearman | Sélection gloutonne (section 4, étape 5) |
| F10 | Exemple de groupe redondant : `Debt ratio %`, `Net worth/Assets`, `Equity to Liability`, `Liability to Equity`, `Total debt/Total net worth` (\|ρ\| entre 0,99 et 1,00) | Matrice de Spearman | Ne garder **qu'une** variable par groupe |

---

## 2. Explorations restant à faire (faits uniquement, sans modifier `data`)

### E1. Les 24 colonnes avec des valeurs autour de 10¹⁰ — prioritaire

Question : erreurs de saisie, codes spéciaux (division par zéro ?), ou vraie échelle brute ?

```python
for col in not_same_scale.index:
    n = (data[col] > 1).sum()
    n_faillite = data.loc[data[col] > 1, "Bankrupt?"].sum()
    print(f"{col:45s} {n:5d} lignes > 1   dont faillites : {n_faillite}")
```

Conclusions possibles à noter en markdown :
- **Quelques lignes seulement** → valeurs aberrantes. Décision côté modélisation : écrêter (`clip`), ou remplacer par NaN puis imputer par la médiane.
- **Beaucoup de lignes** → la colonne n'est probablement pas normalisée. Décision : transformation `log1p` ou `QuantileTransformer`.
- **Surreprésentation des faillites parmi ces lignes** → la valeur extrême est peut-être un signal. Décision : ajouter une colonne indicatrice `col_extreme = (col > 1)` avant de traiter la valeur.

Visualiser aussi la distribution **sans** les valeurs extrêmes (`data.loc[data[col] <= 1, col].hist()`).

### E2. Lien entre chaque variable et la cible

Jusqu'ici on a seulement étudié les variables entre elles. Il faut savoir lesquelles **séparent les faillites des autres**.

```python
from sklearn.feature_selection import mutual_info_classif

X = data.drop(columns=["Bankrupt?", "Net Income Flag"])
y = data["Bankrupt?"]
mi = pd.Series(mutual_info_classif(X, y, random_state=0), index=X.columns).sort_values(ascending=False)
mi.head(15)
```

Puis des boxplots par classe pour les 5 à 10 variables en tête :

```python
for col in mi.index[:8]:
    sns.boxplot(data=data, x="Bankrupt?", y=col, showfliers=False)
    plt.show()
```

> Dans l'exploration, calculer l'information mutuelle sur **toutes** les données est acceptable, puisqu'on ne fait que constater. Dans le notebook de modélisation, la **sélection** des variables devra être refaite **sur le train uniquement** (voir section 4).

L'estimation de `mutual_info_classif` est bruitée sur des variables continues : on peut faire la moyenne sur plusieurs `random_state` pour vérifier la stabilité du classement.

### E3. Vérifications rapides

| Vérification | Code | Ce qu'on cherche |
|---|---|---|
| Lignes en double | `data.duplicated().sum()` | Si > 0 : à supprimer en modélisation |
| Autre variable binaire | `pd.crosstab(data["Liability-Assets Flag"], data["Bankrupt?"])` | Effectif de chaque modalité et taux de faillite associé ; si la modalité 1 est très rare mais toujours en faillite, c'est un signal fort |
| Variables quasi constantes | `data.std().sort_values().head(15)` | Variables comme `Operating Profit Rate` (≈ 0,999 partout) : variance presque nulle, donc peu d'information |
| Pourcentage de faillites | `data["Bankrupt?"].mean() * 100` | Version plus simple de la cellule actuelle |

### E4. (Optionnel) Information mutuelle normalisée entre variables

Pour repérer des dépendances **non monotones** que Spearman ne voit pas (voir section 3).

```python
from sklearn.metrics import normalized_mutual_info_score

# Discrétisation en déciles (rank d'abord pour gérer les valeurs répétées)
Xd = data.drop(columns=["Bankrupt?", "Net Income Flag"]).rank(method="first").apply(lambda c: pd.qcut(c, 10, labels=False))

cols = Xd.columns
nmi = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        nmi.iloc[i, j] = nmi.iloc[j, i] = normalized_mutual_info_score(Xd.iloc[:, i], Xd.iloc[:, j])
```

Ce sont environ 4500 paires, ce qui prend quelques secondes. On réutilise ensuite le même code `triu` + `stack` qu'avec Spearman, et on regarde les paires avec une NMI élevée mais un \|Spearman\| faible.

### E5. Cellule finale « Conclusions »

Terminer le notebook d'exploration par une cellule markdown qui résume les faits F1 à F10 et les résultats de E1 à E4. C'est elle que le notebook de modélisation suivra.

---

## 3. À retenir : corrélation de Spearman vs information mutuelle

### Ce qui va dans le même sens

**Un \|Spearman\| proche de 1 implique une information mutuelle élevée.** Si $X_j$ est une fonction monotone de $X_i$, connaître $X_i$ donne $X_j$ sans incertitude : $H(X_j \mid X_i) = 0$, donc $I = H(X_j)$, la valeur maximale possible.

Comme Spearman, **l'information mutuelle est invariante aux transformations monotones**. Elle ne dépend que des probabilités, pas des valeurs elles-mêmes. La normalisation min-max ne la perturbe donc pas, et elle ne se fait pas piéger par la forme en $1/x$ comme Pearson.

On doit donc retrouver le groupe `Debt ratio %`, `Net worth/Assets`, `Equity to Liability`, etc.

### Ce qui diffère

**1. L'information mutuelle détecte plus de dépendances.** Elle mesure n'importe quelle dépendance, pas seulement les relations monotones. Exemple : $Y = X^2$ avec $X$ symétrique autour de 0. Spearman vaut alors environ 0, car $Y$ monte puis redescend quand $X$ augmente. Pourtant l'information mutuelle est maximale, puisque $X$ détermine $Y$. On a donc :

$$|\rho| \approx 1 \;\Rightarrow\; I \text{ élevée}, \qquad \text{mais} \qquad I \text{ élevée} \;\not\Rightarrow\; |\rho| \approx 1$$

L'information mutuelle peut signaler des paires que Spearman ne voit pas.

**2. Elle n'a ni signe ni échelle fixe.** Elle est en bits et vaut au plus $\min(H(X_i), H(X_j))$. Une variable très dispersée a une entropie élevée, donc une information mutuelle potentiellement plus grande. Les valeurs brutes ne sont donc pas comparables d'une paire à l'autre, contrairement à ρ qui reste toujours entre −1 et 1. Il faut une version normalisée, par exemple :

$$\text{NMI} = \frac{I(X;Y)}{\sqrt{H(X)\,H(Y)}} \in [0,1]$$

**3. La formule du cours est pour des variables discrètes, alors que les $X_i$ sont continues.** Pour des variables continues liées par une fonction exacte, l'information mutuelle théorique est même **infinie**. En pratique, on a deux options :
- **discrétiser**, par exemple en déciles, puis appliquer la formule. Le résultat dépend du nombre de classes ;
- **estimer** avec `sklearn.feature_selection.mutual_info_regression`, qui utilise une méthode des plus proches voisins. L'estimation est bruitée et non normalisée.

### Où l'information mutuelle est vraiment utile

Pour la redondance entre variables, Spearman est plus simple et suffisant. L'information mutuelle est surtout intéressante **entre chaque $X_i$ et la cible `Bankrupt?`**. La cible est discrète, donc la formule s'applique directement, et cela permet de classer les variables par pouvoir prédictif, y compris quand la relation n'est pas monotone.

Démarche retenue : dans chaque groupe de variables redondantes, on **garde celle qui a la plus forte information mutuelle avec `Bankrupt?`**.

---

## 4. Feuille de route pour le notebook de modélisation

L'ordre compte : tout ce qui utilise la cible ou apprend quelque chose des données (sélection, imputation, mise à l'échelle, rééchantillonnage) se fait **après le split et sur le train uniquement**, pour éviter la fuite de données (*data leakage*).

| Étape | Action | Origine |
|---|---|---|
| 1 | `read_csv`, puis `data.columns = data.columns.str.strip()` | F1 |
| 2 | Supprimer `Net Income Flag` et les doublons éventuels | F4, E3 |
| 3 | `train_test_split(..., stratify=y, random_state=...)` | F7 |
| 4 | Traiter les valeurs aberrantes des 24 colonnes, selon la décision prise en E1 (seuils et médianes calculés **sur le train**) | F5, E1 |
| 5 | Sélection des variables : algorithme glouton (ci-dessous), **sur le train** | F8, F9, F10, E2 |
| 6 | Rééquilibrage : `class_weight="balanced"`, ou SMOTE **sur le train uniquement** | F7 |
| 7 | Métriques : rappel, F1, PR-AUC, matrice de confusion. **Pas d'accuracy** (prédire toujours « pas de faillite » donne 96,8 %) | F7 |
| 8 | Modèles : régression logistique (référence, avec `StandardScaler`) → Random Forest → gradient boosting | — |
| 9 | Ajuster le seuil de décision (pas forcément 0,5) selon le compromis rappel / précision voulu | F7 |

### Sélection gloutonne des variables (étape 5)

Raisonner **par groupe** et non paire par paire : avec une chaîne A~B (0,95), B~C (0,95), A~C (0,85), le résultat d'un traitement paire par paire dépend de l'ordre. L'algorithme glouton évite ce problème :
1. trier les variables par information mutuelle avec la cible, de la plus grande à la plus petite ;
2. les parcourir dans cet ordre et garder une variable seulement si elle n'est pas trop corrélée à une variable **déjà gardée**.

```python
from sklearn.feature_selection import mutual_info_classif

mi = pd.Series(mutual_info_classif(X_train, y_train, random_state=0), index=X_train.columns).sort_values(ascending=False)
corr = X_train.corr(method="spearman").abs()

seuil = 0.9
gardees = []
for col in mi.index:                       # de la plus informative à la moins informative
    if all(corr.loc[col, g] < seuil for g in gardees):
        gardees.append(col)

X_train, X_test = X_train[gardees], X_test[gardees]
```

Points d'attention :
- **Seuil** : 0,9 est un choix à justifier ou à tester (0,8 / 0,9 / 0,95), en comparant les performances.
- **Scores proches** : si deux variables d'un groupe ont une information mutuelle très proche, l'écart n'est pas significatif. Garder alors la plus **interprétable** (par exemple `Debt ratio %` plutôt que `Liability to Equity`).
- **Selon le modèle** : cette sélection est importante pour la régression logistique, où la colinéarité rend les coefficients instables. Elle compte moins pour Random Forest et gradient boosting, mais réduit quand même le bruit et le temps de calcul.

---

## 5. Remarques et astuces

### 5.1 Aplatir une matrice avec `stack()`

**Problème rencontré :** filtrer un **DataFrame** avec un masque booléen (`Spearman_corr[Spearman_corr < -0.7]`) ne supprime rien. Le résultat garde la même forme 96 × 96 et met NaN là où le masque vaut `False`, comme `df.where(masque)`. Une grille 2D ne peut pas « perdre une case » sans laisser de trou.

**Solution :** passer en 1D avec `stack()`, qui transforme la matrice en **Series** indexée par le couple `(ligne, colonne)` (un MultiIndex). Au passage, `stack()` **supprime les NaN**. Sur une Series, un masque booléen retire réellement les éléments.

Exemple sur une matrice 3 × 3 :

```
        A      B      C                 A  A    1.00
A    1.00  -0.80   0.30    .stack()     A  B   -0.80
B   -0.80   1.00  -0.95    ───────►     A  C    0.30
C    0.30  -0.95   1.00                 B  A   -0.80
                                        ...    (9 lignes)
```

**Version sans doublons ni diagonale :** la matrice de corrélation est symétrique, donc (A, B) et (B, A) donnent la même valeur, et la diagonale vaut toujours 1. On ne garde que le triangle supérieur strict :

```python
mask = np.triu(np.ones(Spearman_corr.shape, dtype=bool), k=1)   # True au-dessus de la diagonale, k=1 exclut la diagonale
paires = Spearman_corr.where(mask).stack()                      # where met NaN hors du triangle, stack les retire
paires[paires.abs() > 0.9].sort_values(key=abs, ascending=False)
```

- `.where(mask)` utilise volontairement le comportement « NaN au lieu de supprimer », puisque `stack()` élimine ensuite ces NaN.
- `paires.abs() > 0.9` capte les corrélations fortes **dans les deux sens** ; une corrélation de −0,99 indique autant de redondance qu'une corrélation de +0,99.
- `sort_values(key=abs, ascending=False)` trie selon la valeur absolue, mais affiche les valeurs d'origine avec leur signe.
- Pour n colonnes, on passe de n² cases à n(n−1)/2 paires (96 colonnes → 4560 paires).

Pour lire les paires une par une :

```python
for (col1, col2), r in paires[paires.abs() > 0.9].items():
    print(f"{col1}  ↔  {col2} : {r:.3f}")
```

### 5.2 Corrélation de Pearson vs corrélation de Spearman

| | Pearson | Spearman |
|---|---|---|
| Ce qu'elle mesure | Une relation **linéaire** (en ligne droite) | Une relation **monotone** : quand l'une monte, l'autre monte (ou descend) toujours |
| Calcul | Sur les valeurs elles-mêmes | Sur les **rangs** des valeurs (c'est une corrélation de Pearson calculée sur les rangs) |
| Relation $y = 1/x$ | Faible : l'hyperbole est mal approchée par une droite | ≈ −1 : l'ordre est parfaitement inversé |
| Sensibilité aux valeurs extrêmes | Forte : quelques points peuvent tout changer | Faible : une valeur extrême n'est qu'un rang de plus |
| Invariante aux transformations monotones (log, min-max, 1/x…) | Seulement aux transformations affines | Oui |
| Code pandas | `.corr()` (par défaut), `np.corrcoef` | `.corr(method="spearman")` |

**Exemple observé dans ce dataset :** entre `Equity to Liability` et `Liability to Equity`, qui sont l'inverse l'une de l'autre, Pearson donne **−0,16** et Spearman **−0,993**. Un nuage de points montre la forme d'hyperbole :

```python
sns.scatterplot(data=data, x="Equity to Liability", y="Liability to Equity", s=5)
```

**À retenir :**
- `.corr()` et `np.corrcoef` calculent **Pearson** par défaut. Il faut demander Spearman explicitement.
- Pour repérer des variables redondantes entre ratios financiers (souvent liés par un inverse ou un $1 - x$), utiliser **Spearman**.
- Une corrélation faible ne signifie pas une indépendance : Pearson rate les relations non linéaires, Spearman rate les relations non monotones (comme $y = x^2$). L'information mutuelle (section 3) détecte les deux.
- Une corrélation de +1 et une de −1 indiquent la même redondance : on raisonne en **valeur absolue** pour la sélection.

### 5.3 Reconnaître des données déjà normalisées

La documentation d'un dataset ne précise pas toujours si les données ont été transformées (c'est le cas ici : la page UCI n'en parle pas). On peut le déduire avec `describe()` et un peu de bon sens métier :

| Indice | Ce qu'on observe ici | Pourquoi c'est suspect |
|---|---|---|
| **Bornes exactes** | min = 0 et max = 1 sur 72 colonnes sur 96 | Des grandeurs réelles variées ne tombent pas toutes pile entre 0 et 1, avec des extrêmes qui touchent exactement 0 et 1 : c'est la signature d'une normalisation **min-max** $x' = \frac{x - \min}{\max - \min}$ |
| **Relations mathématiques cassées** | `Equity to Liability` et `Liability to Equity` sont toutes deux < 1, et leur produit ≠ 1 | Si $a = 1/b$ et $a < 1$, alors $b > 1$. Une transformation affine appliquée séparément à chaque colonne ne préserve pas la relation d'inverse |
| **Valeurs absurdes métier** | `Operating Profit Rate` ≈ 0,999 pour toutes les entreprises | Une marge opérationnelle de 99,9 % n'existe pas : la valeur a été écrasée vers le haut par quelques extrêmes très bas |
| **Colonnes qui font exception** | 24 colonnes avec un max autour de 10¹⁰ | Elles n'ont pas subi la même transformation, ou contiennent des valeurs aberrantes : à investiguer séparément (E1) |

Autres signatures utiles à connaître :
- **Standardisation** (`StandardScaler`) : moyenne ≈ 0 et écart-type ≈ 1 sur toutes les colonnes, avec des valeurs négatives.
- **Transformation log** : des variables qui devraient être très asymétriques (chiffre d'affaires, actifs) ont une distribution presque symétrique.

**Conséquences :**
- Les valeurs ne s'interprètent plus comme de vrais ratios, mais **l'ordre entre les entreprises est conservé** (la min-max est croissante). Les méthodes basées sur les rangs (Spearman, arbres de décision, information mutuelle) ne sont donc pas affectées.
- Pour vérifier une relation théorique entre deux colonnes (un inverse, une somme égale à 1…), tester la **monotonie** (Spearman) plutôt que l'égalité exacte.
- Le mentionner dans le rapport : « Les variables semblent normalisées min-max sur [0, 1], ce qu'on déduit de leurs bornes et de l'incohérence entre ratios inverses. » La source la plus fiable reste l'article d'origine cité sur la page UCI (Liang et al., 2016, *European Journal of Operational Research*).
