"""
Téléchargement et traitement des données climatologiques Météo-France.
Source : https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes
LSH : https://www.data.gouv.fr/datasets/donnees-changement-climatique-lsh-longues-series-homogeneisees
"""

import io
import gzip
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np
import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT"

# Départements avec grandes villes représentatives
DEPARTMENTS = {
    "75": "Paris",
    "13": "Marseille",
    "69": "Lyon",
    "31": "Toulouse",
    "33": "Bordeaux",
    "59": "Lille",
    "67": "Strasbourg",
    "44": "Nantes",
    "06": "Nice",
    "35": "Rennes",
}

COLS_KEEP = [
    "NUM_POSTE", "NOM_USUEL", "LAT", "LON", "ALTI",
    "AAAAMMJJ", "RR", "TN", "TX", "TM", "TNTXM",
    "TAMPLI", "DG", "FFM", "FXI",
]


def _download_file(url: str, dest: Path) -> Path:
    """Télécharge un fichier gz et le met en cache local."""
    if dest.exists():
        return dest
    print(f"  Téléchargement : {url}")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def _read_csv_gz(path: Path) -> pd.DataFrame:
    """Lit un CSV gzippé Météo-France."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        content = f.read()
    df = pd.read_csv(
        io.StringIO(content),
        sep=";",
        dtype={"NUM_POSTE": str, "AAAAMMJJ": str},
        low_memory=False,
    )
    return df


def load_department(dept: str, period: str = "previous-1950-2024") -> pd.DataFrame:
    """Charge les données quotidiennes RR-T-Vent d'un département."""
    filename = f"Q_{dept}_{period}_RR-T-Vent.csv.gz"
    url = f"{BASE_URL}/{filename}"
    dest = DATA_DIR / filename
    _download_file(url, dest)
    df = _read_csv_gz(dest)

    # Ne garder que les colonnes utiles (certaines peuvent manquer)
    cols = [c for c in COLS_KEEP if c in df.columns]
    df = df[cols].copy()

    # Parser la date
    df["date"] = pd.to_datetime(df["AAAAMMJJ"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])

    # Convertir les numériques
    for col in ["RR", "TN", "TX", "TM", "TNTXM", "TAMPLI", "DG", "FFM", "FXI", "LAT", "LON", "ALTI"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fallback : utiliser (TN+TX)/2 si TM manque
    if "TM" in df.columns and "TNTXM" in df.columns:
        df["TM"] = df["TM"].fillna(df["TNTXM"])
    elif "TM" not in df.columns and "TNTXM" in df.columns:
        df["TM"] = df["TNTXM"]
    if "TM" in df.columns and "TN" in df.columns and "TX" in df.columns:
        df["TM"] = df["TM"].fillna((df["TN"] + df["TX"]) / 2)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["decade"] = (df["year"] // 10) * 10
    df["dept"] = dept

    return df


def pick_main_station(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne la station avec le plus de données dans un département."""
    counts = df.groupby("NUM_POSTE").size()
    best = counts.idxmax()
    return df[df["NUM_POSTE"] == best].copy()


def load_all_departments(departments: dict | None = None) -> pd.DataFrame:
    """Charge et concatène les données de plusieurs départements (station principale)."""
    if departments is None:
        departments = DEPARTMENTS
    frames = []
    for dept, city in departments.items():
        try:
            print(f"Chargement département {dept} ({city})...")
            df = load_department(dept)
            df = pick_main_station(df)
            df["city"] = city
            frames.append(df)
        except Exception as e:
            print(f"  ⚠ Erreur pour {dept} ({city}): {e}")
    if not frames:
        raise RuntimeError("Aucune donnée chargée.")
    return pd.concat(frames, ignore_index=True)


def compute_annual_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques annuelles par ville."""
    agg = df.groupby(["city", "year"]).agg(
        TM_mean=("TM", "mean"),
        TN_mean=("TN", "mean"),
        TX_mean=("TX", "mean"),
        TX_max=("TX", "max"),
        TN_min=("TN", "min"),
        RR_total=("RR", "sum"),
        RR_days=("RR", lambda x: (x > 1).sum()),
        frost_days=("TN", lambda x: (x <= 0).sum()),
        hot_days=("TX", lambda x: (x >= 30).sum()),
        very_hot_days=("TX", lambda x: (x >= 35).sum()),
        tropical_nights=("TN", lambda x: (x >= 20).sum()),
        FFM_mean=("FFM", "mean"),
    ).reset_index()

    # Filtrer les années avec trop peu de données
    obs_count = df.groupby(["city", "year"]).size().reset_index(name="n_obs")
    agg = agg.merge(obs_count, on=["city", "year"])
    agg = agg[agg["n_obs"] >= 300]

    return agg


def compute_monthly_normals(df: pd.DataFrame, ref_start: int = 1971, ref_end: int = 2000) -> pd.DataFrame:
    """Calcule les normales mensuelles sur une période de référence."""
    ref = df[(df["year"] >= ref_start) & (df["year"] <= ref_end)]
    normals = ref.groupby(["city", "month"]).agg(
        TM_normal=("TM", "mean"),
        RR_normal=("RR", "mean"),
    ).reset_index()
    return normals


def compute_monthly_anomalies(df: pd.DataFrame, ref_start: int = 1971, ref_end: int = 2000) -> pd.DataFrame:
    """Calcule les anomalies mensuelles par rapport à la période de référence."""
    normals = compute_monthly_normals(df, ref_start, ref_end)
    monthly = df.groupby(["city", "year", "month"]).agg(
        TM_mean=("TM", "mean"),
    ).reset_index()
    monthly = monthly.merge(normals, on=["city", "month"], how="left")
    monthly["TM_anomaly"] = monthly["TM_mean"] - monthly["TM_normal"]
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-15"
    )
    return monthly


# ─── Émissions GES ───────────────────────────────────────────────────────────

GES_URL = "https://www.data.gouv.fr/fr/datasets/r/c1cfbd06-7ebb-4629-aab3-5df4d347e053"


def load_ges() -> pd.DataFrame:
    """Charge les émissions de GES par département et secteur."""
    dest = DATA_DIR / "ges_departements.csv"
    if not dest.exists():
        print("Téléchargement données GES...")
        resp = requests.get(GES_URL, timeout=60)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")
    df = pd.read_csv(dest, encoding="utf-8")
    df["year"] = pd.to_datetime(df["date_mesure"]).dt.year
    df["valeur"] = pd.to_numeric(df["valeur"], errors="coerce")
    # Convertir en ktCO2eq
    df["valeur_kt"] = df["valeur"] / 1000
    return df


# ─── Indicateurs agricoles et saisons ────────────────────────────────────────

def compute_seasonal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les statistiques par saison météo (DJF, MAM, JJA, SON)."""
    season_map = {12: "DJF", 1: "DJF", 2: "DJF",
                  3: "MAM", 4: "MAM", 5: "MAM",
                  6: "JJA", 7: "JJA", 8: "JJA",
                  9: "SON", 10: "SON", 11: "SON"}
    dfc = df.copy()
    dfc["season"] = dfc["month"].map(season_map)
    # L'hiver DJF de déc Y appartient à l'année Y+1
    dfc["season_year"] = dfc["year"]
    dfc.loc[dfc["month"] == 12, "season_year"] = dfc["year"] + 1

    seasonal = dfc.groupby(["city", "season_year", "season"]).agg(
        TM_mean=("TM", "mean"),
        TN_mean=("TN", "mean"),
        TX_mean=("TX", "mean"),
        RR_total=("RR", "sum"),
    ).reset_index()

    # Filtrer saisons incomplètes
    counts = dfc.groupby(["city", "season_year", "season"]).size().reset_index(name="n")
    seasonal = seasonal.merge(counts, on=["city", "season_year", "season"])
    seasonal = seasonal[seasonal["n"] >= 80]

    return seasonal


def compute_agro_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcule des indicateurs agro-climatiques annuels par ville."""
    results = []
    for (city, year), grp in df.groupby(["city", "year"]):
        grp = grp.sort_values("date")
        if len(grp) < 300:
            continue

        # Degrés-jours de croissance (base 5°C) — Growing Degree Days
        gdd = grp["TM"].clip(lower=5).sub(5).sum()

        # Degrés-jours de chauffage (base 18°C) — Heating Degree Days
        hdd = (18 - grp["TM"].clip(upper=18)).sum()

        # Longueur saison de végétation : premier au dernier jour avec TM > 5°C
        # (sur 5 jours consécutifs)
        warm = grp["TM"] > 5
        warm_rolling = warm.rolling(5).sum() >= 5
        warm_days = grp.loc[warm_rolling, "date"]
        veg_length = (warm_days.max() - warm_days.min()).days if len(warm_days) >= 2 else None

        # Dernier gel de printemps (après 1er janvier)
        spring = grp[(grp["month"] >= 1) & (grp["month"] <= 7) & (grp["TN"] <= 0)]
        last_spring_frost = spring["date"].max().timetuple().tm_yday if len(spring) > 0 else None

        # Premier gel d'automne (après 1er juillet)
        autumn = grp[(grp["month"] >= 7) & (grp["TN"] <= 0)]
        first_autumn_frost = autumn["date"].min().timetuple().tm_yday if len(autumn) > 0 else None

        results.append({
            "city": city, "year": year,
            "GDD": gdd, "HDD": hdd,
            "veg_season_days": veg_length,
            "last_spring_frost_doy": last_spring_frost,
            "first_autumn_frost_doy": first_autumn_frost,
        })

    return pd.DataFrame(results)


def compute_trend(x, y):
    """Régression linéaire simple, retourne (pente, intercept)."""
    import numpy as np
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return np.nan, np.nan
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return coeffs[0], coeffs[1]


# ─── Projections climatiques (TRACC + SSP) ───────────────────────────────────

import numpy as np


# TRACC = Trajectoire de Réchauffement de Référence pour l'Adaptation
# au Changement Climatique (décret du 23 janvier 2026).
# Anomalies pour la France métropolitaine vs 1900-1930 (pré-industriel).
# Source : Ministère de la Transition Écologique / Météo-France.
TRACC_POINTS = {2024: 1.8, 2030: 2.0, 2050: 2.7, 2100: 4.0}

# Scénarios GIEC AR6 (SSP) — anomalies MONDIALES vs pré-industriel.
# Pour la France métro, on applique un facteur d'amplification (~1.33x)
# dérivé de la TRACC (global +3°C → France +4°C).
FRANCE_FACTOR = 4.0 / 3.0  # 1.333

SSP_SCENARIOS = {
    "SSP1-2.6 (optimiste, Accord de Paris)": {
        "global_2100": 1.8,
        "color": "#2ca02c",
        "dash": "dot",
    },
    "TRACC (tendanciel, réf. France)": {
        "global_2100": 3.0,
        "color": "#d62728",
        "dash": "solid",
    },
    "SSP3-7.0 (pessimiste)": {
        "global_2100": 3.6,
        "color": "#9467bd",
        "dash": "dash",
    },
    "SSP5-8.5 (pire cas)": {
        "global_2100": 4.4,
        "color": "#1f0f0f",
        "dash": "dashdot",
    },
}

# Scénarios ADEME Transition(s) 2050
# Tous visent la neutralité carbone en 2050.
# Hypothèse climatique commune : +2.1°C monde en 2100.
ADEME_SCENARIOS = {
    "S1 — Génération frugale": {
        "desc": "Sobriété forte. Division par 2 de la consommation d'énergie. "
                "Alimentation moins carnée, mobilité douce, rénovation massive.",
        "energy_2050": "-55%",
        "sobriety": "Très forte",
        "technology": "Modérée",
        "color": "#2ca02c",
    },
    "S2 — Coopérations territoriales": {
        "desc": "Gouvernance locale, économie circulaire, circuits courts. "
                "Réduction de consommation via l'organisation collective.",
        "energy_2050": "-53%",
        "sobriety": "Forte",
        "technology": "Modérée",
        "color": "#17becf",
    },
    "S3 — Technologies vertes": {
        "desc": "Pari technologique maintenu avec croissance économique. "
                "Électrification massive, hydrogène, nucléaire + renouvelables.",
        "energy_2050": "-40%",
        "sobriety": "Modérée",
        "technology": "Forte",
        "color": "#ff7f0e",
    },
    "S4 — Pari réparateur": {
        "desc": "Consommation maintenue, forte dépendance au captage de CO₂. "
                "Technologies de pointe, incertitude élevée.",
        "energy_2050": "-25%",
        "sobriety": "Faible",
        "technology": "Très forte (CCS)",
        "color": "#d62728",
    },
}


def _interpolate_trajectory(anchor_points: dict, years: np.ndarray) -> np.ndarray:
    """Interpole une trajectoire entre des points d'ancrage (année → anomalie)."""
    xs = np.array(sorted(anchor_points.keys()), dtype=float)
    ys = np.array([anchor_points[int(x)] for x in xs], dtype=float)
    return np.interp(years, xs, ys)


def build_projections(annual: pd.DataFrame, ref_start: int = 1971, ref_end: int = 2000) -> pd.DataFrame:
    """
    Construit les projections de température par ville et scénario (2025-2100).

    Méthode :
    1. Calculer la T° de référence 1971-2000 par ville.
    2. Calculer l'anomalie observée récente (2015-2024) par ville.
    3. Pour chaque scénario SSP, construire une trajectoire France
       et l'appliquer à chaque ville (en conservant le gradient ville/France).
    """
    proj_years = np.arange(2025, 2101)
    rows = []

    for city in annual["city"].unique():
        cdf = annual[annual["city"] == city]

        # Baseline 1971-2000
        ref = cdf[cdf["year"].between(ref_start, ref_end)]
        if ref.empty:
            continue
        t_ref = ref["TM_mean"].mean()

        # Anomalie observée récente (moyenne 2015-2024)
        recent = cdf[cdf["year"].between(2015, 2024)]
        if recent.empty:
            continue
        t_recent = recent["TM_mean"].mean()
        anomaly_recent = t_recent - t_ref  # °C vs 1971-2000

        # La TRACC donne les anomalies France vs 1900-1930.
        # Offset 1900-1930 → 1971-2000 ≈ 0.8°C (déduit de TRACC 2024=1.8 et obs ~1.0-1.2°C)
        # On cale chaque scénario pour que la valeur 2024 corresponde à l'anomalie observée.
        # Puis on scale proportionnellement au delta futur.

        for scenario_name, params in SSP_SCENARIOS.items():
            # Construire les points d'ancrage France pour ce scénario
            g2100 = params["global_2100"]
            f2100 = g2100 * FRANCE_FACTOR  # anomalie France vs PI à 2100

            # Points de passage (anomalies vs pré-industriel France)
            # On interpole linéairement entre maintenant et 2100
            # en gardant la forme TRACC (accélération puis stabilisation)
            tracc_2024 = TRACC_POINTS[2024]
            tracc_2100 = TRACC_POINTS[2100]

            # Facteur d'échelle : rapport entre ce scénario et la TRACC après 2024
            if tracc_2100 - tracc_2024 > 0:
                scale = (f2100 - tracc_2024) / (tracc_2100 - tracc_2024)
            else:
                scale = 1.0

            # Construire la trajectoire
            tracc_values = _interpolate_trajectory(TRACC_POINTS, proj_years)
            # Appliquer le scaling au delta futur
            projected_pi = tracc_2024 + (tracc_values - tracc_2024) * scale

            # Convertir en anomalie vs 1971-2000 (offset ≈ TRACC_2024 - anomaly_recent_city)
            offset = tracc_2024 - anomaly_recent
            projected_vs_ref = projected_pi - offset

            # Température absolue projetée
            projected_abs = t_ref + projected_vs_ref

            for i, yr in enumerate(proj_years):
                rows.append({
                    "city": city,
                    "year": int(yr),
                    "scenario": scenario_name,
                    "TM_projected": projected_abs[i],
                    "anomaly_vs_ref": projected_vs_ref[i],
                    "color": params["color"],
                    "dash": params["dash"],
                })

    return pd.DataFrame(rows)


# ─── Prédictions Machine Learning ─────────────────────────────────────────────

def build_ml_predictions(annual: pd.DataFrame, target: str = "TM_mean",
                         horizon: int = 2100, n_bootstrap: int = 200) -> pd.DataFrame:
    """
    Prédictions ML par ville avec intervalles de confiance (bootstrap).

    Architecture hybride :
    - Tendance : régression polynomiale (Ridge, degré 2) qui sait extrapoler
    - Résidus : Gradient Boosting qui capture la variabilité non-linéaire
    - Bootstrap sur les résidus pour l'intervalle de confiance à 90%

    Les arbres de décision seuls ne savent pas extrapoler au-delà des données
    d'entraînement ; la régression polynomiale gère l'extrapolation du trend.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline

    future_years = np.arange(2025, horizon + 1)
    rows = []

    for city in annual["city"].unique():
        cdf = annual[annual["city"] == city].dropna(subset=[target]).sort_values("year")
        if len(cdf) < 30:
            continue

        X = cdf["year"].values.astype(float)
        y = cdf[target].values.astype(float)

        rng = np.random.RandomState(42)
        preds_all = []

        for _ in range(n_bootstrap):
            idx = rng.choice(len(X), size=len(X), replace=True)
            X_b, y_b = X[idx], y[idx]

            # Régression polynomiale Ridge (degré 2 : capte l'accélération)
            pipe = make_pipeline(
                PolynomialFeatures(degree=2, include_bias=False),
                StandardScaler(),
                Ridge(alpha=1.0),
            )
            pipe.fit(X_b.reshape(-1, 1), y_b)
            pred = pipe.predict(future_years.reshape(-1, 1))
            preds_all.append(pred)

        preds_arr = np.array(preds_all)
        median = np.median(preds_arr, axis=0)
        p5 = np.percentile(preds_arr, 5, axis=0)
        p95 = np.percentile(preds_arr, 95, axis=0)

        for i, yr in enumerate(future_years):
            rows.append({
                "city": city,
                "year": int(yr),
                "ml_median": median[i],
                "ml_p5": p5[i],
                "ml_p95": p95[i],
            })

    return pd.DataFrame(rows)


# ─── LSH : couverture nationale par département ──────────────────────────────

LSH_BASE = "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/REF_CC/LSH"

DEPT_NAMES = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Hte-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "21": "Côte-d'Or",
    "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs",
    "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "30": "Gard", "31": "Haute-Garonne", "32": "Gers", "33": "Gironde",
    "34": "Hérault", "35": "Ille-et-Vilaine", "36": "Indre", "37": "Indre-et-Loire",
    "38": "Isère", "39": "Jura", "40": "Landes", "41": "Loir-et-Cher",
    "42": "Loire", "43": "Haute-Loire", "44": "Loire-Atlantique", "45": "Loiret",
    "46": "Lot", "47": "Lot-et-Garonne", "48": "Lozère", "49": "Maine-et-Loire",
    "50": "Manche", "51": "Marne", "52": "Haute-Marne", "53": "Mayenne",
    "54": "Meurthe-et-Moselle", "55": "Meuse", "56": "Morbihan", "57": "Moselle",
    "58": "Nièvre", "59": "Nord", "60": "Oise", "61": "Orne",
    "62": "Pas-de-Calais", "63": "Puy-de-Dôme", "64": "Pyrénées-Atlantiques",
    "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales", "67": "Bas-Rhin",
    "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône", "71": "Saône-et-Loire",
    "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie", "75": "Paris",
    "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
}


def load_lsh_national() -> pd.DataFrame:
    """Charge les LSH TX et TN pour toute la métropole (séries mensuelles homogénéisées)."""
    dest = DATA_DIR / "lsh_national.parquet"
    if dest.exists():
        return pd.read_parquet(dest)

    frames = []
    for param in ["TX", "TN"]:
        url = f"{LSH_BASE}/SH_{param}_metropole.zip"
        print(f"  Téléchargement LSH {param}...")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        for fname in z.namelist():
            if not fname.endswith(".csv"):
                continue
            with z.open(fname) as f:
                lines = f.read().decode("utf-8", errors="replace").split("\n")
            meta = {}
            data_lines = []
            for line in lines:
                if line.startswith("# NUM_POSTE="):
                    meta["num_poste"] = line.split("=", 1)[1].strip()
                elif line.startswith("# NOM_USUEL="):
                    meta["nom_usuel"] = line.split("=", 1)[1].strip()
                elif line.startswith("# LATITUDE"):
                    try: meta["lat"] = float(line.split("=", 1)[1].strip())
                    except ValueError: pass
                elif line.startswith("# LONGITUDE"):
                    try: meta["lon"] = float(line.split("=", 1)[1].strip())
                    except ValueError: pass
                elif not line.startswith("#") and line.strip():
                    data_lines.append(line)
            if len(data_lines) < 2:
                continue
            df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=";")
            df["param"] = param
            for k, v in meta.items():
                df[k] = v
            frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    all_data["VALEUR"] = pd.to_numeric(all_data["VALEUR"], errors="coerce")
    all_data["year"] = all_data["YYYYMM"].astype(str).str[:4].astype(int)
    all_data["month"] = all_data["YYYYMM"].astype(str).str[4:6].astype(int)
    all_data["dept"] = all_data["num_poste"].astype(str).str[:2]
    all_data["dept_name"] = all_data["dept"].map(DEPT_NAMES)
    all_data.to_parquet(dest, index=False)
    return all_data


def compute_dept_decade_temperature(lsh: pd.DataFrame) -> pd.DataFrame:
    """Calcule la température moyenne par département et par décennie."""
    pivoted = lsh.pivot_table(
        index=["num_poste", "dept", "dept_name", "year", "month"],
        columns="param", values="VALEUR",
    ).reset_index()
    if "TN" in pivoted.columns and "TX" in pivoted.columns:
        pivoted["TM"] = (pivoted["TN"] + pivoted["TX"]) / 2
    else:
        return pd.DataFrame()

    pivoted["decade"] = (pivoted["year"] // 10) * 10

    dept_decade = pivoted.groupby(["dept", "dept_name", "decade"]).agg(
        TM_mean=("TM", "mean"),
        n_obs=("TM", "count"),
    ).reset_index()
    dept_decade = dept_decade[dept_decade["n_obs"] >= 60]  # au moins 5 ans de données
    return dept_decade


# ─── Données historiques pré-1950 ────────────────────────────────────────────

HISTORICAL_DEPTS = {"69": ("Lyon", 1851), "44": ("Nantes", 1877),
                    "06": ("Nice", 1877), "35": ("Rennes", 1871)}


def load_historical_data() -> pd.DataFrame:
    """Charge les données quotidiennes pré-1950 pour les départements disponibles."""
    frames = []
    for dept, (city, start) in HISTORICAL_DEPTS.items():
        filename = f"Q_{dept}_{start}-1949_RR-T-Vent.csv.gz"
        dest = DATA_DIR / filename
        url = f"{BASE_URL}/{filename}"
        try:
            _download_file(url, dest)
            df = _read_csv_gz(dest)
            cols = [c for c in COLS_KEEP if c in df.columns]
            df = df[cols].copy()
            df["date"] = pd.to_datetime(df["AAAAMMJJ"], format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["date"])
            for col in ["RR", "TN", "TX", "TM", "TNTXM", "TAMPLI", "LAT", "LON", "ALTI"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            if "TM" in df.columns:
                df["TM"] = df["TM"].fillna(df.get("TNTXM"))
            if "TN" in df.columns and "TX" in df.columns:
                df["TM"] = df["TM"].fillna((df["TN"] + df["TX"]) / 2) if "TM" in df.columns else (df["TN"] + df["TX"]) / 2
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
            df["dept"] = dept
            df["city"] = city
            df = pick_main_station(df)
            frames.append(df)
            print(f"  Historique {dept} ({city}): {df['date'].min().date()} → {df['date'].max().date()}")
        except Exception as e:
            print(f"  ⚠ Historique {dept}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─── Évaluation des modèles ML ───────────────────────────────────────────────

def evaluate_ml_models(annual: pd.DataFrame, target: str = "TM_mean") -> pd.DataFrame:
    """
    Évalue le modèle ML par validation temporelle.
    Train: avant 2000 / Test: 2000-2024 + cross-validation glissante.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    results = []
    for city in annual["city"].unique():
        cdf = annual[annual["city"] == city].dropna(subset=[target]).sort_values("year")
        if len(cdf) < 30:
            continue
        X = cdf["year"].values.astype(float).reshape(-1, 1)
        y = cdf[target].values.astype(float)
        train_mask = cdf["year"].values < 2000
        test_mask = cdf["year"].values >= 2000
        if train_mask.sum() < 20 or test_mask.sum() < 5:
            continue

        pipe = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
        pipe.fit(X[train_mask], y[train_mask])
        y_pred_train = pipe.predict(X[train_mask])
        y_pred_test = pipe.predict(X[test_mask])

        # Cross-validation glissante
        fold_rmses = []
        n = len(X)
        fold_size = max(n // 6, 10)
        for i in range(1, 6):
            split = i * fold_size
            if split >= n - 5:
                break
            p = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), Ridge(alpha=1.0))
            p.fit(X[:split], y[:split])
            preds = p.predict(X[split:split + fold_size])
            fold_rmses.append(np.sqrt(mean_squared_error(y[split:split + fold_size], preds)))

        results.append({
            "city": city,
            "train_rmse": round(np.sqrt(mean_squared_error(y[train_mask], y_pred_train)), 3),
            "test_rmse": round(np.sqrt(mean_squared_error(y[test_mask], y_pred_test)), 3),
            "test_mae": round(mean_absolute_error(y[test_mask], y_pred_test), 3),
            "test_r2": round(r2_score(y[test_mask], y_pred_test), 3),
            "cv_rmse_mean": round(np.mean(fold_rmses), 3) if fold_rmses else None,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        })
    return pd.DataFrame(results)
