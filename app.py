"""
Changement Climatique en France — Dashboard Streamlit
=====================================================
Défi : https://defis.data.gouv.fr/defis/changement-climatique
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_loader import (
    DEPARTMENTS,
    load_all_departments,
    compute_annual_stats,
    compute_monthly_anomalies,
    compute_seasonal_stats,
    compute_agro_indicators,
    compute_trend,
    load_ges,
    build_projections,
    build_ml_predictions,
    SSP_SCENARIOS,
    ADEME_SCENARIOS,
    TRACC_POINTS,
    load_lsh_national,
    compute_dept_decade_temperature,
    load_historical_data,
    evaluate_ml_models,
    HISTORICAL_DEPTS,
    DEPT_NAMES,
)

# ─── Config ──────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Climat France", page_icon="🌍", layout="wide")

RED = "#d62728"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
MONTH_NAMES = ["Jan", "Fév", "Mar", "Avr", "Mai", "Jun",
               "Jul", "Aoû", "Sep", "Oct", "Nov", "Déc"]
REF_START, REF_END = 1971, 2000


# ─── Chargement ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Chargement des données Météo-France...")
def get_data():
    raw = load_all_departments()
    annual = compute_annual_stats(raw)
    anomalies = compute_monthly_anomalies(raw)
    seasonal = compute_seasonal_stats(raw)
    agro = compute_agro_indicators(raw)
    projections = build_projections(annual)
    ml_preds = build_ml_predictions(annual)
    return raw, annual, anomalies, seasonal, agro, projections, ml_preds

@st.cache_data(show_spinner="Chargement des émissions GES...")
def get_ges():
    return load_ges()

@st.cache_data(show_spinner="Chargement des données historiques (pré-1950)...")
def get_historical():
    hist = load_historical_data()
    if hist.empty:
        return pd.DataFrame(), pd.DataFrame()
    hist_annual = compute_annual_stats(hist)
    return hist, hist_annual

@st.cache_data(show_spinner="Chargement des LSH (couverture nationale)...")
def get_lsh():
    lsh = load_lsh_national()
    dept_decade = compute_dept_decade_temperature(lsh)
    return lsh, dept_decade

@st.cache_data(show_spinner="Évaluation des modèles ML...")
def get_ml_eval(annual):
    return evaluate_ml_models(annual)

@st.cache_data(show_spinner="Chargement du GeoJSON des départements...")
def get_geojson():
    import requests as _req
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson"
    resp = _req.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()

raw, annual, anomalies, seasonal, agro, projections, ml_preds = get_data()
ges = get_ges()
hist_raw, hist_annual = get_historical()

cities = sorted(annual["city"].unique())
year_min, year_max = int(annual["year"].min()), int(annual["year"].max())


# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.title("Paramètres")
selected_cities = st.sidebar.multiselect("Villes", cities, default=["Paris", "Marseille", "Lyon"])
year_range = st.sidebar.slider("Période d'observation", year_min, year_max, (1960, year_max))

st.sidebar.markdown("---")
st.sidebar.caption(
    "[Données Météo-France](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes) · "
    "[Émissions GES CITEPA](https://www.data.gouv.fr/datasets/emissions-de-gaz-a-effet-de-serre-annuelles-par-secteur) · "
    "[TRACC](https://www.ecologie.gouv.fr/politiques-publiques/"
    "trajectoire-rechauffement-reference-ladaptation-changement-climatique-tracc) · "
    "Licence Ouverte 2.0"
)

# Filtres
df_a = annual[annual["city"].isin(selected_cities) & annual["year"].between(*year_range)]
df_anom = anomalies[anomalies["city"].isin(selected_cities) & anomalies["year"].between(*year_range)]
df_raw = raw[raw["city"].isin(selected_cities) & raw["year"].between(*year_range)]
df_season = seasonal[seasonal["city"].isin(selected_cities) & seasonal["season_year"].between(*year_range)]
df_agro = agro[agro["city"].isin(selected_cities) & agro["year"].between(*year_range)]


# ─── En-tête ─────────────────────────────────────────────────────────────────

st.title("Changement climatique en France")
st.caption(
    "Ce tableau de bord vous permet d'explorer 75 ans de données météo en France, "
    "de comprendre comment le climat a déjà changé, et de voir ce que prévoient "
    "les scientifiques et les modèles de machine learning pour les décennies à venir."
)

# KPIs
if not df_a.empty:
    recent = df_a[df_a["year"] >= 2010]
    ref = df_a[df_a["year"].between(REF_START, REF_END)]
    if not recent.empty and not ref.empty:
        c1, c2, c3, c4, c5 = st.columns(5)
        dt = recent["TM_mean"].mean() - ref["TM_mean"].mean()
        c1.metric("Réchauffement observé", f"{dt:+.1f} °C",
                   help=f"Écart entre la moyenne 2010-{year_max} et la normale {REF_START}-{REF_END}")
        dh = recent["hot_days"].mean() - ref["hot_days"].mean()
        c2.metric("Jours > 30°C / an", f"{recent['hot_days'].mean():.0f}", f"{dh:+.0f}")
        df_ = recent["frost_days"].mean() - ref["frost_days"].mean()
        c3.metric("Jours de gel / an", f"{recent['frost_days'].mean():.0f}", f"{df_:+.0f}", delta_color="inverse")
        dn = recent["tropical_nights"].mean() - ref["tropical_nights"].mean()
        c4.metric("Nuits > 20°C / an", f"{recent['tropical_nights'].mean():.0f}", f"{dn:+.0f}")
        slope, _ = compute_trend(df_a["year"].values.astype(float), df_a["TM_mean"].values)
        if not np.isnan(slope):
            c5.metric("Vitesse", f"{slope*10:+.2f} °C/déc.",
                       help="Le climat se réchauffe de ce nombre de degrés tous les 10 ans")


# ─── Onglets ─────────────────────────────────────────────────────────────────

tab_synth, tab_temp, tab_anom, tab_ext, tab_pluie, tab_saison, tab_agro, \
    tab_ges, tab_proj, tab_ml, tab_ademe, tab_stripes, tab_carte, tab_choro = st.tabs([
    "Synthèse",
    "Températures",
    "Anomalies",
    "Extrêmes",
    "Pluie",
    "Saisons",
    "Agriculture",
    "Émissions CO2",
    "Projections 2100",
    "Prédiction ML",
    "Scénarios ADEME",
    "Warming Stripes",
    "Carte stations",
    "Carte départements",
])


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHÈSE
# ══════════════════════════════════════════════════════════════════════════════

with tab_synth:
    st.header("En résumé : ce qu'il faut retenir")

    st.info(
        "**La France se réchauffe 1,5 à 2 fois plus vite que la moyenne mondiale.** "
        "Depuis 1950, la température moyenne a augmenté d'environ **1,8°C** sur le territoire "
        "(+1,9°C de réchauffement anthropique selon Météo-France en 2025), "
        "avec une accélération nette depuis les années 2000."
    )

    st.markdown("### Ce qui a déjà changé (observations réelles)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "- Les **étés sont plus chauds** : les canicules (> 35°C) sont "
            "**2 à 3 fois plus fréquentes** qu'avant 2000\n"
            "- Les **hivers sont plus doux** : le nombre de jours de gel "
            "a chuté d'environ 30%\n"
            "- Les **nuits restent chaudes** en été : les nuits tropicales "
            "(> 20°C) se multiplient, surtout en ville"
        )
    with col2:
        st.markdown(
            "- Les **vendanges ont avancé de 23 jours** depuis les années 1970\n"
            "- La **mer monte de 3,7 mm par an** (Chiffres clés du climat 2025)\n"
            "- Les **glaciers alpins** ont perdu 36 m d'épaisseur "
            "d'eau depuis 2001\n"
            "- **2022 et 2023** restent les deux années les plus chaudes "
            "jamais enregistrées en France (2025 = 4e, 2024 = 5e)"
        )

    st.markdown("### Ce qui nous attend")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Trajectoire officielle France (TRACC)")
        st.markdown(
            "| Horizon | Réchauffement France |\n"
            "|---------|---------------------|\n"
            "| **2030** | **+2,0°C** vs pré-industriel |\n"
            "| **2050** | **+2,7°C** |\n"
            "| **2100** | **+4,0°C** |\n\n"
            "*La TRACC est inscrite dans la loi française depuis janvier 2026.*"
        )
    with col4:
        st.markdown("#### En pratique, ça veut dire quoi ?")
        st.markdown(
            "- **+2°C (2030)** : Étés comme 2003 tous les 5 ans. "
            "Stress hydrique récurrent dans le Sud.\n"
            "- **+2,7°C (2050)** : Canicules de 2019 = un été « normal ». "
            "Paris aura le climat actuel de Lyon.\n"
            "- **+4°C (2100)** : Bordeaux aura le climat de Séville. "
            "Certaines cultures impossibles dans le Sud."
        )

    st.warning(
        "**Chaque dixième de degré compte.** Selon le GIEC, les impacts à +2°C sont "
        "bien plus graves qu'à +1,5°C. Le budget carbone pour rester sous +1,5°C "
        "est quasiment épuisé : il correspond à environ **3 ans** d'émissions au rythme actuel (2025)."
    )

    st.caption(
        "Sources : [TRACC](https://www.ecologie.gouv.fr/politiques-publiques/"
        "trajectoire-rechauffement-reference-ladaptation-changement-climatique-tracc) · "
        "[Chiffres clés climat 2025](https://www.statistiques.developpement-durable.gouv.fr/"
        "edition-numerique/chiffres-cles-du-climat/fr/3-observations-du-changement-climatique-et) · "
        "[Bon Pote — GIEC](https://bonpote.com/le-rapport-du-giec-pour-les-parents-et-enseignants/) · "
        "[Our World in Data](https://ourworldindata.org/climate-change)"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEMPÉRATURES
# ══════════════════════════════════════════════════════════════════════════════

with tab_temp:
    st.header("Évolution des températures")
    st.caption(
        "Chaque point représente la température moyenne d'une année. "
        "La courbe lissée montre la tendance de fond, au-delà des variations naturelles."
    )

    # Fusionner données historiques pré-1950 pour les villes disponibles
    plot_annual = df_a.copy()
    if not hist_annual.empty:
        hist_cities = [c for c in selected_cities if c in hist_annual["city"].values]
        if hist_cities:
            hist_sel = hist_annual[hist_annual["city"].isin(hist_cities)]
            # Éviter les doublons d'années
            existing = set(zip(plot_annual["city"], plot_annual["year"]))
            hist_new = hist_sel[~hist_sel.apply(lambda r: (r["city"], r["year"]) in existing, axis=1)]
            if not hist_new.empty:
                plot_annual = pd.concat([hist_new, plot_annual], ignore_index=True).sort_values(["city", "year"])
                st.info(
                    f"Données historiques pré-1950 disponibles pour : "
                    f"**{', '.join(hist_cities)}** (séries remontant au XIXe siècle)."
                )

    fig = px.scatter(plot_annual, x="year", y="TM_mean", color="city", trendline="lowess",
                     labels={"year": "Année", "TM_mean": "Température moyenne (°C)", "city": "Ville"},
                     height=500)
    fig.update_traces(marker=dict(size=4, opacity=0.5))
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    # Tendances
    st.markdown("#### Vitesse du réchauffement par ville")
    st.caption("La colonne « °C/décennie » indique de combien la température monte tous les 10 ans.")
    trends = []
    for city in selected_cities:
        cdf = df_a[df_a["city"] == city]
        if len(cdf) < 10:
            continue
        s, i = compute_trend(cdf["year"].values.astype(float), cdf["TM_mean"].values)
        if not np.isnan(s):
            trends.append({"Ville": city, "°C / décennie": round(s * 10, 3),
                           "Température projetée 2050 (tendance linéaire)": f"{i + s * 2050:.1f}°C"})
    if trends:
        st.dataframe(pd.DataFrame(trends), hide_index=True, use_container_width=True)

    # Enveloppe
    st.markdown("#### Records annuels : minimum la nuit (bleu) vs maximum le jour (rouge)")
    st.caption("La zone grisée montre l'écart entre les extrêmes de chaque année.")
    for city in selected_cities:
        cdf = df_a[df_a["city"] == city]
        if cdf.empty:
            continue
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=cdf["year"], y=cdf["TX_max"], mode="lines", name="Max jour", line=dict(color=RED)))
        fig2.add_trace(go.Scatter(x=cdf["year"], y=cdf["TN_min"], mode="lines", name="Min nuit",
                                  line=dict(color=BLUE), fill="tonexty", fillcolor="rgba(200,200,200,0.15)"))
        fig2.update_layout(title=city, height=280, xaxis_title="", yaxis_title="°C",
                           legend=dict(orientation="h", y=-0.25), margin=dict(t=35))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANOMALIES
# ══════════════════════════════════════════════════════════════════════════════

with tab_anom:
    st.header("Anomalies de température")
    st.caption(
        f"Chaque barre montre l'écart de température d'un mois par rapport à la « normale » "
        f"({REF_START}-{REF_END}). Rouge = plus chaud que la normale. Bleu = plus froid."
    )

    # Anomalie annuelle agrégée
    yearly_anom = df_anom.groupby("year")["TM_anomaly"].mean().reset_index()
    colors_y = [RED if v > 0 else BLUE for v in yearly_anom["TM_anomaly"]]
    fig_ya = go.Figure(go.Bar(x=yearly_anom["year"], y=yearly_anom["TM_anomaly"], marker_color=colors_y))
    fig_ya.update_layout(height=350, xaxis_title="Année", yaxis_title="Écart vs normale (°C)",
                         title="Anomalie annuelle moyenne (villes sélectionnées)")
    st.plotly_chart(fig_ya, use_container_width=True)

    st.markdown(
        "> On voit clairement que **les barres rouges dominent depuis les années 1990** : "
        "presque chaque année est plus chaude que la normale du XXe siècle."
    )

    # Heatmap
    st.markdown("#### Carte de chaleur : quel mois de quelle année a été le plus anormal ?")
    city_hm = st.selectbox("Ville", selected_cities, key="hm_city")
    hm_data = df_anom[df_anom["city"] == city_hm].pivot_table(index="month", columns="year", values="TM_anomaly")
    if not hm_data.empty:
        fig_hm = px.imshow(hm_data, color_continuous_scale="RdBu_r", origin="lower",
                           labels={"x": "Année", "y": "Mois", "color": "Écart (°C)"},
                           y=[MONTH_NAMES[i] for i in range(len(hm_data))],
                           aspect="auto", height=380)
        st.plotly_chart(fig_hm, use_container_width=True)
        st.caption("Les cases rouge foncé montrent les mois avec le réchauffement le plus marqué.")


# ══════════════════════════════════════════════════════════════════════════════
# EXTRÊMES
# ══════════════════════════════════════════════════════════════════════════════

with tab_ext:
    st.header("Événements extrêmes")
    st.caption(
        "Le changement climatique ne se résume pas à « il fait un peu plus chaud ». "
        "Ce sont surtout les extrêmes qui augmentent : canicules, nuits étouffantes, "
        "et en parallèle, les jours de gel disparaissent."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Jours de canicule (> 35°C)")
        fig = px.bar(df_a, x="year", y="very_hot_days", color="city", barmode="group",
                     labels={"year": "Année", "very_hot_days": "Nombre de jours", "city": "Ville"},
                     color_discrete_sequence=px.colors.qualitative.Set2, height=380)
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("#### Nuits étouffantes (min > 20°C)")
        st.caption("Quand la température ne descend pas sous 20°C la nuit, le corps ne récupère pas.")
        fig = px.bar(df_a, x="year", y="tropical_nights", color="city", barmode="group",
                     labels={"year": "Année", "tropical_nights": "Nombre de nuits", "city": "Ville"},
                     color_discrete_sequence=px.colors.qualitative.Set2, height=380)
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Jours de gel (min < 0°C)")
        st.caption("Moins de gel = moins de régulation des parasites, arbres fruitiers perturbés.")
        fig = px.bar(df_a, x="year", y="frost_days", color="city", barmode="group",
                     labels={"year": "Année", "frost_days": "Nombre de jours", "city": "Ville"},
                     color_discrete_sequence=px.colors.qualitative.Set2, height=380)
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.markdown("#### Records de température")
        records = []
        for city in selected_cities:
            cdf = df_raw[df_raw["city"] == city]
            if cdf.empty:
                continue
            idx_max = cdf["TX"].idxmax()
            idx_min = cdf["TN"].idxmin()
            if pd.notna(idx_max) and pd.notna(idx_min):
                records.append({"Ville": city,
                                "Record chaleur": f"{cdf.loc[idx_max, 'TX']:.1f}°C ({cdf.loc[idx_max, 'date'].strftime('%d/%m/%Y')})",
                                "Record froid": f"{cdf.loc[idx_min, 'TN']:.1f}°C ({cdf.loc[idx_min, 'date'].strftime('%d/%m/%Y')})"})
        if records:
            st.dataframe(pd.DataFrame(records), hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLUIE
# ══════════════════════════════════════════════════════════════════════════════

with tab_pluie:
    st.header("Précipitations")
    st.caption(
        "Le changement climatique modifie aussi le régime des pluies : "
        "les événements intenses augmentent tandis que la répartition saisonnière change."
    )

    fig = px.scatter(df_a, x="year", y="RR_total", color="city", trendline="lowess",
                     labels={"year": "Année", "RR_total": "Total annuel (mm)", "city": "Ville"}, height=420)
    fig.update_traces(marker=dict(size=4, opacity=0.5))
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Saisonnalité : comment la pluie se répartit-elle dans l'année ?")
    st.caption("Comparez les décennies : la répartition mensuelle des pluies évolue-t-elle ?")
    city_precip = st.selectbox("Ville", selected_cities, key="precip_city")
    cdf = df_raw[(df_raw["city"] == city_precip) & (df_raw["decade"] >= 1960)]
    if not cdf.empty:
        monthly_rr = cdf.groupby(["decade", "month"])["RR"].mean().reset_index()
        fig = px.line(monthly_rr, x="month", y="RR", color="decade",
                      labels={"month": "Mois", "RR": "Pluie moyenne (mm/jour)", "decade": "Décennie"},
                      height=380, color_discrete_sequence=px.colors.sequential.YlOrRd)
        fig.update_layout(xaxis=dict(tickmode="array", tickvals=list(range(1, 13)), ticktext=MONTH_NAMES),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SAISONS
# ══════════════════════════════════════════════════════════════════════════════

with tab_saison:
    st.header("Évolution par saison")
    st.caption(
        "Toutes les saisons ne se réchauffent pas à la même vitesse. "
        "La courbe fine montre les données brutes, la courbe épaisse la moyenne sur 10 ans."
    )

    season_names = {"DJF": "Hiver", "MAM": "Printemps", "JJA": "Été", "SON": "Automne"}
    season_colors = {"DJF": "#1f77b4", "MAM": "#2ca02c", "JJA": "#d62728", "SON": "#ff7f0e"}

    for city in selected_cities:
        cdf = df_season[df_season["city"] == city]
        if cdf.empty:
            continue
        fig = go.Figure()
        for s in ["DJF", "MAM", "JJA", "SON"]:
            sdf = cdf[cdf["season"] == s].sort_values("season_year")
            if sdf.empty:
                continue
            fig.add_trace(go.Scatter(x=sdf["season_year"], y=sdf["TM_mean"], mode="lines",
                                    name=season_names[s], line=dict(color=season_colors[s], width=1), opacity=0.4))
            if len(sdf) >= 10:
                r = sdf.set_index("season_year")["TM_mean"].rolling(10, center=True).mean()
                fig.add_trace(go.Scatter(x=r.index, y=r.values, mode="lines",
                                        name=f"{season_names[s]} (tendance)", line=dict(color=season_colors[s], width=3)))
        fig.update_layout(title=city, height=380, xaxis_title="", yaxis_title="°C",
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=35))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Quel mois se réchauffe le plus vite ?")
    month_trends = []
    for city in selected_cities:
        for m in range(1, 13):
            mdf = df_anom[(df_anom["city"] == city) & (df_anom["month"] == m)]
            if len(mdf) < 20:
                continue
            s, _ = compute_trend(mdf["year"].values.astype(float), mdf["TM_mean"].values)
            if not np.isnan(s):
                month_trends.append({"Ville": city, "Mois": MONTH_NAMES[m - 1], "°C/décennie": round(s * 10, 2)})
    if month_trends:
        mt = pd.DataFrame(month_trends)
        fig = px.bar(mt, x="Mois", y="°C/décennie", color="Ville", barmode="group",
                     height=380, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(xaxis=dict(categoryorder="array", categoryarray=MONTH_NAMES),
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Les mois avec les barres les plus hautes sont ceux qui se réchauffent le plus vite.")


# ══════════════════════════════════════════════════════════════════════════════
# AGRICULTURE
# ══════════════════════════════════════════════════════════════════════════════

with tab_agro:
    st.header("Impact sur l'agriculture")
    st.caption(
        "Le climat contrôle directement ce qu'on peut cultiver et quand. "
        "Voici des indicateurs clés pour l'agriculture française."
    )

    st.markdown(
        "| Indicateur | Ce que ça mesure |\n"
        "|-----------|------------------|\n"
        "| **Degrés-jours de croissance** (GDD) | Énergie thermique dispo pour les plantes (base 5°C). Plus c'est haut, plus ça pousse vite. |\n"
        "| **Degrés-jours de chauffage** (HDD) | Besoin de chauffer les bâtiments (base 18°C). Moins = hivers plus doux. |\n"
        "| **Saison de végétation** | Nombre de jours où les plantes peuvent pousser (T > 5°C). |\n"
        "| **Dernier gel de printemps** | Après cette date, plus de risque de gel pour les cultures. |"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Énergie thermique pour les cultures (GDD)")
        fig = px.scatter(df_agro, x="year", y="GDD", color="city", trendline="lowess",
                         labels={"year": "Année", "GDD": "Degrés-jours (base 5°C)", "city": "Ville"}, height=380)
        fig.update_traces(marker=dict(size=4, opacity=0.5))
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La hausse des GDD permet des cultures plus méridionales mais augmente le stress hydrique.")
    with col2:
        st.markdown("#### Besoin de chauffage (HDD)")
        fig = px.scatter(df_agro, x="year", y="HDD", color="city", trendline="lowess",
                         labels={"year": "Année", "HDD": "Degrés-jours (base 18°C)", "city": "Ville"}, height=380)
        fig.update_traces(marker=dict(size=4, opacity=0.5))
        fig.update_layout(legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La baisse des HDD signifie qu'on a besoin de moins chauffer l'hiver — bonne nouvelle pour l'énergie.")

    st.markdown("#### Durée de la saison de végétation")
    fig = px.scatter(df_agro.dropna(subset=["veg_season_days"]),
                     x="year", y="veg_season_days", color="city", trendline="lowess",
                     labels={"year": "Année", "veg_season_days": "Jours de végétation", "city": "Ville"}, height=380)
    fig.update_traces(marker=dict(size=4, opacity=0.5))
    fig.update_layout(legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("La saison s'allonge : les plantes démarrent plus tôt au printemps et continuent plus tard à l'automne.")


# ══════════════════════════════════════════════════════════════════════════════
# ÉMISSIONS GES
# ══════════════════════════════════════════════════════════════════════════════

with tab_ges:
    st.header("Émissions de gaz à effet de serre")
    st.caption(
        "Les gaz à effet de serre (CO2, méthane...) sont la cause du réchauffement. "
        "Voici les émissions des départements correspondant aux villes analysées."
    )

    our_depts = list(DEPARTMENTS.keys())
    ges_ours = ges[ges["geocode_departement"].isin(our_depts)].copy()
    ges_ours["city"] = ges_ours["geocode_departement"].map(DEPARTMENTS)
    latest_year = ges_ours["year"].max()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### Qui émet le plus ? ({latest_year})")
        ges_total = ges_ours[ges_ours["year"] == latest_year].groupby("city")["valeur_kt"].sum().reset_index()
        ges_total = ges_total.sort_values("valeur_kt", ascending=True)
        fig = px.bar(ges_total, x="valeur_kt", y="city", orientation="h",
                     labels={"valeur_kt": "Émissions (ktCO2eq)", "city": ""},
                     color="valeur_kt", color_continuous_scale="YlOrRd", height=380)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"#### Quels secteurs ? ({latest_year})")
        ges_sector = ges_ours[ges_ours["year"] == latest_year].groupby("secteur")["valeur_kt"].sum().reset_index()
        fig = px.pie(ges_sector, values="valeur_kt", names="secteur", height=380,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("Les transports et l'industrie sont les plus gros émetteurs. L'agriculture pèse aussi lourd via le méthane.")

    # Croisement
    st.markdown("#### Réchauffement local vs émissions du département")
    st.caption("Les villes qui émettent le plus ne sont pas forcément celles qui se réchauffent le plus, "
               "car le réchauffement est un phénomène global.")
    cross = []
    for city in cities:
        cref = annual[(annual["city"] == city) & annual["year"].between(REF_START, REF_END)]
        crec = annual[(annual["city"] == city) & (annual["year"] >= 2010)]
        if cref.empty or crec.empty:
            continue
        delta = crec["TM_mean"].mean() - cref["TM_mean"].mean()
        ges_c = ges_ours[(ges_ours["city"] == city) & (ges_ours["year"] == latest_year)]
        if ges_c.empty:
            continue
        cross.append({"Ville": city, "Réchauffement (°C)": round(delta, 2),
                       "Émissions (ktCO2eq)": round(ges_c["valeur_kt"].sum(), 0)})
    if cross:
        cdf = pd.DataFrame(cross)
        fig = px.scatter(cdf, x="Réchauffement (°C)", y="Émissions (ktCO2eq)", text="Ville",
                         size="Émissions (ktCO2eq)", height=420,
                         color="Réchauffement (°C)", color_continuous_scale="RdYlBu_r")
        fig.update_traces(textposition="top center")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PROJECTIONS 2100 (scénarios GIEC/TRACC)
# ══════════════════════════════════════════════════════════════════════════════

with tab_proj:
    st.header("Projections des scientifiques jusqu'en 2100")
    st.caption(
        "Ces courbes montrent les trajectoires possibles selon les scénarios du GIEC. "
        "La partie gauche (trait plein noir) = données réelles. "
        "La partie droite = projections selon les politiques climatiques adoptées."
    )

    st.markdown(
        "| Scénario | Hypothèse | Réchauffement France 2100 |\n"
        "|----------|-----------|---------------------------|\n"
        "| **SSP1-2.6** | Accord de Paris respecté | ~+2,4°C |\n"
        "| **TRACC** | Politiques actuelles maintenues (réf. officielle France) | **+4,0°C** |\n"
        "| **SSP3-7.0** | Politiques insuffisantes | ~+4,8°C |\n"
        "| **SSP5-8.5** | Développement 100% fossile | ~+5,9°C |"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("France 2030 (TRACC)", f"+{TRACC_POINTS[2030]:.1f}°C")
    c2.metric("France 2050 (TRACC)", f"+{TRACC_POINTS[2050]:.1f}°C")
    c3.metric("France 2100 (TRACC)", f"+{TRACC_POINTS[2100]:.1f}°C")

    for city in selected_cities:
        obs = df_a[df_a["city"] == city][["year", "TM_mean"]].sort_values("year")
        proj_city = projections[projections["city"] == city]
        if obs.empty or proj_city.empty:
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=obs["year"], y=obs["TM_mean"], mode="lines",
                                 name="Observations", line=dict(color="#333", width=2)))
        for sc_name, params in SSP_SCENARIOS.items():
            sdf = proj_city[proj_city["scenario"] == sc_name].sort_values("year")
            fig.add_trace(go.Scatter(x=sdf["year"], y=sdf["TM_projected"], mode="lines",
                                     name=sc_name, line=dict(color=params["color"], width=2, dash=params["dash"])))
        fig.add_vline(x=2024, line_dash="dot", line_color="grey", annotation_text="Aujourd'hui")
        fig.update_layout(title=city, height=420, xaxis_title="", yaxis_title="Température moyenne (°C)",
                          legend=dict(orientation="h", y=-0.2), xaxis=dict(range=[year_range[0], 2105]),
                          margin=dict(t=35))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRÉDICTION ML
# ══════════════════════════════════════════════════════════════════════════════

with tab_ml:
    st.header("Prédiction par Machine Learning")

    st.info(
        "**Comment ça marche ?** Un modèle de Gradient Boosting (une technique d'intelligence artificielle) "
        "apprend la tendance de réchauffement à partir des 75 ans de données réelles, "
        "puis extrapole jusqu'en 2100. On répète l'exercice 200 fois avec des échantillons "
        "légèrement différents (bootstrap) pour obtenir un **intervalle de confiance à 90%** : "
        "la zone colorée montre où la température a 90% de chances de se trouver."
    )

    st.markdown(
        "| | Scénarios GIEC (onglet précédent) | Prédiction ML (cet onglet) |\n"
        "|---|---|---|\n"
        "| **Méthode** | Modèles physiques du climat mondial | Apprentissage sur données locales |\n"
        "| **Atout** | Intègre la physique du climat, les émissions | Capte la dynamique locale réelle |\n"
        "| **Limite** | Résolution spatiale grossière | Ne « connaît » pas la physique du climat |\n"
        "| **Utilité** | Planification politique | Vérification, tendance locale |"
    )

    for city in selected_cities:
        obs = df_a[df_a["city"] == city][["year", "TM_mean"]].sort_values("year")
        ml = ml_preds[ml_preds["city"] == city].sort_values("year")
        tracc = projections[(projections["city"] == city) &
                            (projections["scenario"].str.contains("TRACC"))].sort_values("year")
        if obs.empty or ml.empty:
            continue

        fig = go.Figure()
        # Intervalle de confiance ML
        fig.add_trace(go.Scatter(
            x=pd.concat([ml["year"], ml["year"][::-1]]),
            y=pd.concat([ml["ml_p95"], ml["ml_p5"][::-1]]),
            fill="toself", fillcolor="rgba(255,127,14,0.15)",
            line=dict(width=0), showlegend=True, name="ML — intervalle 90%",
        ))
        # Médiane ML
        fig.add_trace(go.Scatter(x=ml["year"], y=ml["ml_median"], mode="lines",
                                 name="ML — prédiction médiane",
                                 line=dict(color="#ff7f0e", width=3)))
        # TRACC pour comparaison
        if not tracc.empty:
            fig.add_trace(go.Scatter(x=tracc["year"], y=tracc["TM_projected"], mode="lines",
                                     name="TRACC (scénario officiel)",
                                     line=dict(color=RED, width=2, dash="dash")))
        # Observations
        fig.add_trace(go.Scatter(x=obs["year"], y=obs["TM_mean"], mode="lines",
                                 name="Observations réelles",
                                 line=dict(color="#333", width=2)))
        fig.add_vline(x=2024, line_dash="dot", line_color="grey", annotation_text="Aujourd'hui")
        fig.update_layout(title=city, height=450, xaxis_title="", yaxis_title="Température moyenne (°C)",
                          legend=dict(orientation="h", y=-0.2), xaxis=dict(range=[year_range[0], 2105]),
                          margin=dict(t=35))
        st.plotly_chart(fig, use_container_width=True)

    # Tableau comparatif ML vs TRACC
    st.markdown("#### Comparaison ML vs TRACC par ville")
    comp_rows = []
    for city in selected_cities:
        ml_c = ml_preds[(ml_preds["city"] == city)]
        tr_c = projections[(projections["city"] == city) & projections["scenario"].str.contains("TRACC")]
        row = {"Ville": city}
        for h in [2050, 2075, 2100]:
            mlr = ml_c[ml_c["year"] == h]
            trr = tr_c[tr_c["year"] == h]
            if not mlr.empty:
                row[f"ML {h}"] = f"{mlr['ml_median'].iloc[0]:.1f}°C [{mlr['ml_p5'].iloc[0]:.1f} — {mlr['ml_p95'].iloc[0]:.1f}]"
            if not trr.empty:
                row[f"TRACC {h}"] = f"{trr['TM_projected'].iloc[0]:.1f}°C"
        comp_rows.append(row)
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

    # Évaluation du modèle
    st.markdown("---")
    st.markdown("#### Évaluation du modèle (validation temporelle)")
    st.caption(
        "Le modèle est entraîné sur les données **avant 2000**, puis testé sur les années **2000-2024** "
        "(qu'il n'a jamais vues). Cela permet de vérifier s'il capture bien la tendance réelle."
    )
    ml_eval = get_ml_eval(annual)
    if not ml_eval.empty:
        eval_display = ml_eval.copy()
        eval_display.columns = [
            "Ville", "RMSE entraînement (°C)", "RMSE test (°C)", "MAE test (°C)",
            "R² test", "RMSE CV glissante (°C)", "N entraînement", "N test"
        ]
        st.dataframe(eval_display, hide_index=True, use_container_width=True)

        st.markdown(
            "| Métrique | Signification |\n"
            "|----------|---------------|\n"
            "| **RMSE** | Erreur moyenne quadratique — plus c'est bas, mieux c'est (en °C) |\n"
            "| **MAE** | Erreur moyenne absolue — interprétation directe en °C |\n"
            "| **R²** | Part de la variance expliquée — 1.0 = parfait, 0 = inutile |\n"
            "| **CV glissante** | Validation croisée temporelle (5 fenêtres) — mesure la robustesse |"
        )

        avg_r2 = ml_eval["test_r2"].mean()
        avg_rmse = ml_eval["test_rmse"].mean()
        if avg_r2 > 0.5:
            st.success(
                f"Le modèle explique en moyenne **{avg_r2:.0%}** de la variabilité des températures "
                f"sur la période test (2000-2024), avec une erreur moyenne de **{avg_rmse:.2f}°C**."
            )
        else:
            st.info(
                f"R² moyen sur le test : {avg_r2:.2f}. Le signal tendanciel est capté, "
                f"mais la variabilité interannuelle reste difficile à prédire (ce qui est normal)."
            )

    st.warning(
        "**Attention** : les prédictions ML au-delà de 2050 sont de plus en plus incertaines "
        "(l'intervalle s'élargit). Le ML ne remplace pas les modèles climatiques physiques du GIEC — "
        "il les complète en donnant une perspective « data-driven » locale."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCÉNARIOS ADEME
# ══════════════════════════════════════════════════════════════════════════════

with tab_ademe:
    st.header("Les 4 chemins vers la neutralité carbone (ADEME 2050)")
    st.caption(
        "L'ADEME propose 4 scénarios contrastés pour atteindre zéro émission nette en 2050 en France. "
        "Tous les 4 y parviennent, mais par des voies très différentes."
    )

    for name, p in ADEME_SCENARIOS.items():
        with st.container():
            st.markdown(f"### {name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Consommation d'énergie 2050", p["energy_2050"], help="vs 2015")
            c2.metric("Niveau de sobriété", p["sobriety"])
            c3.metric("Pari technologique", p["technology"])
            st.markdown(f"_{p['desc']}_")
            st.divider()

    # Radar
    st.markdown("#### Comparaison visuelle")
    ademe_df = pd.DataFrame([
        {"Scénario": n.split("—")[0].strip(),
         "Réduction énergie": abs(int(p["energy_2050"].replace("%", "").strip())),
         "Sobriété": {"Faible": 1, "Modérée": 2, "Forte": 3, "Très forte": 4}[p["sobriety"]],
         "Technologie": {"Modérée": 2, "Forte": 3, "Très forte (CCS)": 4}[p["technology"]],
         "color": p["color"]}
        for n, p in ADEME_SCENARIOS.items()
    ])
    fig = go.Figure()
    cats = ["Réduction énergie", "Sobriété", "Technologie"]
    for _, r in ademe_df.iterrows():
        vals = [r["Réduction énergie"]/55*100, r["Sobriété"]/4*100, r["Technologie"]/4*100]  # normalisé sur S1 (max réduction)
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill="toself",
                                      name=r["Scénario"], line=dict(color=r["color"]), opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      height=420, legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, use_container_width=True)

    st.success(
        "**Le message clé** : la neutralité carbone est possible en 2050, "
        "mais elle demande des transformations profondes dans tous les scénarios. "
        "Plus on mise sur la sobriété (S1, S2), moins on dépend de technologies "
        "risquées comme le captage de CO2 (S4)."
    )
    st.caption("Source : [ADEME — Transition(s) 2050](https://www.ademe.fr/les-futurs-en-transition/les-scenarios/)")


# ══════════════════════════════════════════════════════════════════════════════
# WARMING STRIPES
# ══════════════════════════════════════════════════════════════════════════════

with tab_stripes:
    st.header("Warming Stripes")
    st.caption(
        "Visualisation inventée par le climatologue Ed Hawkins. "
        "Chaque bande verticale = 1 année. Bleu = plus froid que la moyenne, "
        "rouge = plus chaud. Aucun chiffre, juste l'évidence visuelle du réchauffement."
    )

    for city in selected_cities:
        cdf = df_a[df_a["city"] == city].sort_values("year")
        if cdf.empty:
            continue
        mean_t = cdf["TM_mean"].mean()
        std_t = cdf["TM_mean"].std()
        fig = go.Figure()
        for _, row in cdf.iterrows():
            val = max(-2.5, min(2.5, (row["TM_mean"] - mean_t) / std_t))
            t = (val + 2.5) / 5.0
            r = int(min(255, 50 + t * 205))
            g = int(max(0, 100 - abs(t - 0.5) * 200))
            b = int(min(255, 255 - t * 205))
            fig.add_trace(go.Bar(x=[row["year"]], y=[1], marker_color=f"rgb({r},{g},{b})",
                                 showlegend=False,
                                 hovertemplate=f"{int(row['year'])}: {row['TM_mean']:.1f}°C<extra></extra>"))
        fig.update_layout(title=city, height=160, bargap=0, xaxis=dict(title=""),
                          yaxis=dict(visible=False), margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARTE
# ══════════════════════════════════════════════════════════════════════════════

with tab_carte:
    st.header("Carte des stations")
    st.caption("Chaque cercle représente une station météo. La taille indique le nombre d'observations.")

    stations = df_raw.groupby(["city", "NOM_USUEL", "NUM_POSTE"]).agg(
        LAT=("LAT", "first"), LON=("LON", "first"), ALTI=("ALTI", "first"),
        n_obs=("date", "count")).reset_index()
    fig = px.scatter_map(stations, lat="LAT", lon="LON", hover_name="NOM_USUEL",
                         hover_data={"city": True, "ALTI": True, "n_obs": True},
                         color="city", size="n_obs", size_max=20,
                         zoom=4.5, center={"lat": 46.5, "lon": 2.5}, height=500)
    fig.update_layout(legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Température par décennie")
    available_decades = sorted(df_raw["decade"].unique())
    if available_decades:
        decade_sel = st.select_slider("Décennie", options=available_decades, value=available_decades[-1])
        cd = df_raw[df_raw["decade"] == decade_sel].groupby("city").agg(
            TM_mean=("TM", "mean"), LAT=("LAT", "first"), LON=("LON", "first")).reset_index()
        if not cd.empty:
            fig = px.scatter_map(cd, lat="LAT", lon="LON", hover_name="city",
                                 color="TM_mean", size=[20]*len(cd), color_continuous_scale="RdYlBu_r",
                                 zoom=4.5, center={"lat": 46.5, "lon": 2.5}, height=420,
                                 labels={"TM_mean": "T° moy (°C)"})
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# CARTE CHOROPLÈTHE PAR DÉPARTEMENT
# ══════════════════════════════════════════════════════════════════════════════

with tab_choro:
    st.header("Carte du réchauffement par département")
    st.caption(
        "Cette carte montre l'évolution de la température moyenne par département français, "
        "calculée à partir des 321 stations des Longues Séries Homogénéisées (LSH) de Météo-France. "
        "Déplacez le curseur pour comparer les décennies."
    )

    try:
        lsh, dept_decade = get_lsh()
        geojson = get_geojson()

        if not dept_decade.empty:
            available_decades = sorted(dept_decade["decade"].unique())
            if len(available_decades) >= 2:
                dec_col1, dec_col2 = st.columns(2)
                with dec_col1:
                    decade_choro = st.select_slider(
                        "Décennie à afficher",
                        options=available_decades,
                        value=available_decades[-1],
                        key="choro_decade",
                    )
                with dec_col2:
                    decade_ref = st.select_slider(
                        "Décennie de référence (pour l'anomalie)",
                        options=available_decades,
                        value=available_decades[2] if len(available_decades) > 2 else available_decades[0],
                        key="choro_ref",
                    )

                # Température absolue de la décennie choisie
                dd_sel = dept_decade[dept_decade["decade"] == decade_choro].copy()

                # Anomalie vs référence
                dd_ref = dept_decade[dept_decade["decade"] == decade_ref][["dept", "TM_mean"]].rename(
                    columns={"TM_mean": "TM_ref"}
                )
                dd_sel = dd_sel.merge(dd_ref, on="dept", how="left")
                dd_sel["anomaly"] = dd_sel["TM_mean"] - dd_sel["TM_ref"]

                # Carte température absolue
                st.markdown(f"#### Température moyenne — décennie {decade_choro}")
                fig_abs = px.choropleth_map(
                    dd_sel,
                    geojson=geojson,
                    locations="dept",
                    featureidkey="properties.code",
                    color="TM_mean",
                    color_continuous_scale="RdYlBu_r",
                    hover_name="dept_name",
                    hover_data={"TM_mean": ":.1f", "dept": False},
                    labels={"TM_mean": "T° moy (°C)"},
                    center={"lat": 46.5, "lon": 2.5},
                    zoom=4.5,
                    height=520,
                )
                fig_abs.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_abs, use_container_width=True)

                # Carte anomalie
                dd_anom = dd_sel.dropna(subset=["anomaly"])
                if not dd_anom.empty:
                    st.markdown(f"#### Anomalie : décennie {decade_choro} vs {decade_ref}")
                    max_abs = max(abs(dd_anom["anomaly"].min()), abs(dd_anom["anomaly"].max()), 0.5)
                    fig_anom = px.choropleth_map(
                        dd_anom,
                        geojson=geojson,
                        locations="dept",
                        featureidkey="properties.code",
                        color="anomaly",
                        color_continuous_scale="RdBu_r",
                        range_color=(-max_abs, max_abs),
                        hover_name="dept_name",
                        hover_data={"anomaly": ":.2f", "dept": False},
                        labels={"anomaly": "Anomalie (°C)"},
                        center={"lat": 46.5, "lon": 2.5},
                        zoom=4.5,
                        height=520,
                    )
                    fig_anom.update_layout(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig_anom, use_container_width=True)

                    st.caption(
                        "Les départements en rouge se sont réchauffés par rapport à la décennie de référence, "
                        "ceux en bleu se sont refroidis (rare)."
                    )
        else:
            st.warning("Données LSH non disponibles.")
    except Exception as e:
        st.error(f"Erreur lors du chargement des données LSH : {e}")


# ─── Footer ──────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "[Données Météo-France](https://www.data.gouv.fr/datasets/donnees-climatologiques-de-base-quotidiennes) · "
    "[Émissions GES CITEPA](https://www.data.gouv.fr/datasets/emissions-de-gaz-a-effet-de-serre-annuelles-par-secteur) · "
    "[TRACC](https://www.ecologie.gouv.fr/politiques-publiques/"
    "trajectoire-rechauffement-reference-ladaptation-changement-climatique-tracc) · "
    "[GIEC AR6](https://www.ipcc.ch/report/ar6/wg1/) · "
    "[ADEME Transition(s) 2050](https://www.ademe.fr/les-futurs-en-transition/les-scenarios/) · "
    "[Bon Pote](https://bonpote.com/le-rapport-du-giec-pour-les-parents-et-enseignants/) · "
    "[Our World in Data](https://ourworldindata.org/climate-change) · "
    "[Chiffres clés climat 2025](https://www.statistiques.developpement-durable.gouv.fr/"
    "edition-numerique/chiffres-cles-du-climat/fr/3-observations-du-changement-climatique-et) · "
    "Licence Ouverte 2.0 · "
    "[Défi](https://defis.data.gouv.fr/defis/changement-climatique)"
)
