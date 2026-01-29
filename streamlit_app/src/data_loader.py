"""
Chargement des données et du modèle ML pour Optiplan
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from models import Troncon, Scenario, ParametresCouts, ParametresSeuils, RegleCout

# Chemins
BASE_DIR = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
SCENARIOS_DIR = Path(__file__).parent.parent / "data" / "scenarios"
PARAMS_FILE = Path(__file__).parent.parent / "data" / "parametres.json"


def load_model():
    """Charge le modèle ML."""
    # Essayer différents chemins de modèle
    model_paths = [
        ARTIFACTS_DIR / "model_v3_final.joblib",
        ARTIFACTS_DIR / "best_model.joblib",
        ARTIFACTS_DIR / "model_h1.joblib",
    ]

    for model_path in model_paths:
        if model_path.exists():
            artifact = joblib.load(model_path)
            print(f"Modèle chargé: {model_path.name}")
            return artifact

    raise FileNotFoundError(f"Aucun modèle trouvé dans {ARTIFACTS_DIR}")


def load_patrimoine_raw() -> pd.DataFrame:
    """Charge les données brutes du patrimoine."""
    # Essayer le dataset préparé avec features
    pkl_path = ARTIFACTS_DIR / "dataset_freeze_h1.pkl"
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
        # Filtrer uniquement les tronçons en service
        if "statut" in df.columns:
            df = df[df["statut"] == "EN SERVICE"].copy()
        elif "STATUT_OBJET" in df.columns:
            df = df[df["STATUT_OBJET"] == "EN SERVICE"].copy()
        return df

    # Sinon charger le CSV brut
    csv_path = DATA_DIR / "v1_trafic_prepared.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["STATUT_OBJET"] == "EN SERVICE"].copy()
        return df

    raise FileNotFoundError("Données patrimoine non trouvées")


def score_troncons(df: pd.DataFrame, model_artifact: dict) -> np.ndarray:
    """Score les tronçons avec le modèle ML."""
    model = model_artifact.get("model")
    calibrator = model_artifact.get("calibrator")
    features = model_artifact.get("features", [])

    if model is None:
        # Fallback: utiliser l'âge normalisé comme proxy
        if "age_at_freeze" in df.columns:
            age = df["age_at_freeze"].values
        elif "age" in df.columns:
            age = df["age"].values / 365.25  # Convertir jours en années
        else:
            return np.random.uniform(0, 0.3, len(df))

        # Normaliser: 0-100 ans -> 0-1
        scores = np.clip(age / 100, 0, 1)
        return scores

    # Préparer les features
    available_features = [f for f in features if f in df.columns]

    if len(available_features) == 0:
        # Pas de features disponibles, utiliser fallback
        print("Warning: Aucune feature ML disponible, utilisation du fallback basé sur l'âge")
        if "age_at_freeze" in df.columns:
            return np.clip(df["age_at_freeze"].values / 100, 0, 1)
        return np.random.uniform(0, 0.3, len(df))

    X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Prédire
    try:
        scores_raw = model.predict_proba(X)[:, 1]
        if calibrator is not None:
            scores = calibrator.transform(scores_raw)
        else:
            scores = scores_raw
        return scores
    except Exception as e:
        print(f"Erreur scoring ML: {e}")
        # Fallback
        if "age_at_freeze" in df.columns:
            return np.clip(df["age_at_freeze"].values / 100, 0, 1)
        return np.random.uniform(0, 0.3, len(df))


def load_troncons(params_couts: Optional[ParametresCouts] = None) -> List[Troncon]:
    """Charge tous les tronçons avec leurs scores ML."""
    print("Chargement des données...")

    # Charger données brutes
    df = load_patrimoine_raw()
    print(f"  {len(df):,} tronçons en service")

    # Charger modèle
    try:
        model_artifact = load_model()
    except FileNotFoundError:
        print("  Warning: Modèle non trouvé, utilisation de scores basés sur l'âge")
        model_artifact = {}

    # Scorer
    scores = score_troncons(df, model_artifact)
    print(f"  Scores ML calculés (mean={scores.mean():.3f})")

    # Paramètres de coûts par défaut
    if params_couts is None:
        params_couts = load_parametres_couts()

    # Mapper les colonnes selon le format du DataFrame
    col_gid = "GID" if "GID" in df.columns else "gid"
    # Gérer les cas où MAT a été renommé (MAT_x après jointure)
    if "MAT" in df.columns:
        col_mat = "MAT"
    elif "MAT_x" in df.columns:
        col_mat = "MAT_x"
    elif "materiau" in df.columns:
        col_mat = "materiau"
    else:
        col_mat = None
    col_diam = "DIAMETRE" if "DIAMETRE" in df.columns else "diametre"
    col_lng = "LNG" if "LNG" in df.columns else "longueur"
    col_annee = "DDP_year" if "DDP_year" in df.columns else "annee_pose"
    col_statut = "STATUT_OBJET" if "STATUT_OBJET" in df.columns else "statut"
    col_age = "age" if "age" in df.columns else "age_at_freeze"

    # Extraire les arrays pour construction rapide
    gids = df[col_gid].astype(str).values if col_gid in df.columns else np.arange(len(df)).astype(str)
    materiaux = df[col_mat].astype(str).values if col_mat and col_mat in df.columns else np.full(len(df), "INCONNU")
    diametres = df[col_diam].fillna(100).values if col_diam in df.columns else np.full(len(df), 100.0)
    longueurs = df[col_lng].fillna(0).values if col_lng in df.columns else np.zeros(len(df))
    annees = df[col_annee].fillna(1970).astype(int).values if col_annee in df.columns else np.full(len(df), 1970)
    statuts = df[col_statut].astype(str).values if col_statut in df.columns else np.full(len(df), "EN SERVICE")
    ages = df[col_age].fillna(0).astype(int).values if col_age in df.columns else np.zeros(len(df), dtype=int)

    # Scores opportunité aléatoires
    np.random.seed(42)
    scores_opp = np.random.uniform(0, 0.5, len(df))

    # Construire les objets Troncon
    troncons = []
    for idx in range(len(df)):
        materiau = materiaux[idx]
        diametre = float(diametres[idx])
        cout_unitaire = params_couts.get_cout(materiau, diametre)

        troncon = Troncon(
            gid=gids[idx],
            materiau=materiau,
            diametre=diametre,
            longueur=float(longueurs[idx]),
            annee_pose=int(annees[idx]),
            statut=statuts[idx],
            age_jours=int(ages[idx]),
            score_ml=float(scores[idx]) if idx < len(scores) else 0.0,
            score_opportunite=float(scores_opp[idx]),
            cout_unitaire=cout_unitaire,
        )
        troncons.append(troncon)

    print(f"  {len(troncons):,} tronçons chargés")
    return troncons


def get_lineaire_total(troncons: List[Troncon]) -> float:
    """Retourne le linéaire total en km."""
    return sum(t.longueur for t in troncons) / 1000


# ============================================================================
# SCENARIOS
# ============================================================================

def ensure_scenarios_dir():
    """Crée le dossier scenarios s'il n'existe pas."""
    SCENARIOS_DIR.mkdir(parents=True, exist_ok=True)


def save_scenario(scenario: Scenario) -> None:
    """Sauvegarde un scénario en JSON."""
    ensure_scenarios_dir()
    filepath = SCENARIOS_DIR / f"{scenario.id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(scenario.to_dict(), f, indent=2, ensure_ascii=False)


def load_scenario(scenario_id: str) -> Optional[Scenario]:
    """Charge un scénario depuis JSON."""
    filepath = SCENARIOS_DIR / f"{scenario_id}.json"
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Scenario.from_dict(data)


def list_scenarios() -> List[Scenario]:
    """Liste tous les scénarios sauvegardés."""
    ensure_scenarios_dir()
    scenarios = []
    for filepath in SCENARIOS_DIR.glob("*.json"):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            scenarios.append(Scenario.from_dict(data))
        except Exception as e:
            print(f"Erreur lecture {filepath}: {e}")
    # Trier par date de modification décroissante
    scenarios.sort(key=lambda s: s.date_modification, reverse=True)
    return scenarios


def delete_scenario(scenario_id: str) -> bool:
    """Supprime un scénario."""
    filepath = SCENARIOS_DIR / f"{scenario_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


# ============================================================================
# PARAMETRES
# ============================================================================

def load_parametres() -> Tuple[ParametresCouts, ParametresSeuils]:
    """Charge les paramètres depuis le fichier JSON."""
    if PARAMS_FILE.exists():
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        couts = ParametresCouts.from_dict(data.get("couts", {}))
        seuils = ParametresSeuils.from_dict(data.get("seuils", {}))
    else:
        couts = get_default_couts()
        seuils = ParametresSeuils()
    return couts, seuils


def load_parametres_couts() -> ParametresCouts:
    """Charge uniquement les paramètres de coûts."""
    couts, _ = load_parametres()
    return couts


def load_parametres_seuils() -> ParametresSeuils:
    """Charge uniquement les paramètres de seuils."""
    _, seuils = load_parametres()
    return seuils


def save_parametres(couts: ParametresCouts, seuils: ParametresSeuils) -> None:
    """Sauvegarde les paramètres."""
    PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "couts": couts.to_dict(),
        "seuils": seuils.to_dict(),
    }
    with open(PARAMS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_default_couts() -> ParametresCouts:
    """Retourne les coûts par défaut."""
    return ParametresCouts(
        cout_defaut=150.0,
        regles=[
            RegleCout(materiau="FT", diametre_min=0, diametre_max=150, cout_unitaire=220),
            RegleCout(materiau="FT", diametre_min=151, diametre_max=300, cout_unitaire=350),
            RegleCout(materiau="FT", diametre_min=301, diametre_max=9999, cout_unitaire=500),
            RegleCout(materiau="FTG", diametre_min=0, diametre_max=150, cout_unitaire=220),
            RegleCout(materiau="FTG", diametre_min=151, diametre_max=300, cout_unitaire=350),
            RegleCout(materiau="PVC", diametre_min=0, diametre_max=150, cout_unitaire=180),
            RegleCout(materiau="PVC", diametre_min=151, diametre_max=300, cout_unitaire=250),
            RegleCout(materiau="PEHD", diametre_min=0, diametre_max=150, cout_unitaire=140),
            RegleCout(materiau="PEHD", diametre_min=151, diametre_max=300, cout_unitaire=200),
            RegleCout(materiau="AC", diametre_min=0, diametre_max=9999, cout_unitaire=280),
        ]
    )
