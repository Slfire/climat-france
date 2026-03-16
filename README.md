# Changement climatique en France

Dashboard interactif pour explorer 75 ans de données climatiques en France, avec projections jusqu'en 2100.

Réalisé dans le cadre du [défi data.gouv.fr](https://defis.data.gouv.fr/defis/changement-climatique).

## Aperçu

14 onglets : synthèse, températures (depuis 1851), anomalies, extrêmes, pluie, saisons, agriculture, émissions CO2, projections 2100 (TRACC + SSP), prédiction ML, scénarios ADEME, warming stripes, carte des stations, carte choroplèthe par département.

## Installation et lancement

```bash
# 1. Cloner le repo
git clone https://github.com/Slfire/climat-france.git
cd climat-france

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le dashboard
streamlit run app.py
```

Le dashboard s'ouvre automatiquement dans le navigateur sur http://localhost:8501.

Au premier lancement, les données sont téléchargées automatiquement depuis les serveurs Météo-France (~200 Mo). Elles sont ensuite mises en cache dans le dossier `data/`.

## Sources de données

| Source | Description |
|--------|-------------|
| [Météo-France — Données quotidiennes](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes) | Températures, pluie, vent (10 départements) |
| [Météo-France — LSH](https://www.data.gouv.fr/datasets/donnees-changement-climatique-lsh-longues-series-homogeneisees) | Séries homogénéisées (321 stations, depuis 1951) |
| [CITEPA](https://www.data.gouv.fr/datasets/emissions-de-gaz-a-effet-de-serre-annuelles-par-secteur) | Émissions GES par département et secteur |
| [TRACC](https://www.ecologie.gouv.fr/politiques-publiques/trajectoire-rechauffement-reference-ladaptation-changement-climatique-tracc) | Trajectoire officielle France (+4°C en 2100) |
| [GIEC AR6](https://www.ipcc.ch/report/ar6/wg1/) | Scénarios SSP mondiaux |
| [ADEME Transition(s) 2050](https://www.ademe.fr/les-futurs-en-transition/les-scenarios/) | 4 scénarios de neutralité carbone |

## Stack technique

- **Python 3.10+**
- **Streamlit** — interface web interactive
- **Plotly** — graphiques et cartes
- **Pandas / NumPy** — traitement de données
- **scikit-learn** — prédictions ML (Ridge polynomiale + bootstrap)
- **statsmodels** — courbes de tendance LOWESS

## Licence

Données sous [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/).
