"""
Configuration globale du projet de priorisation des renouvellements de canalisations.
"""
import os
from pathlib import Path

# Chemins
BASE_DIR = Path(r"c:\Users\ahmed\OneDrive\Bureau\EAU")
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Fichiers sources
TRAFIC_FILE = DATA_DIR / "v1_trafic_prepared.csv"
ANOMALIES_FILE = DATA_DIR / "historiqueanomalie.csv"

# Horizons de prédiction à tester (années)
HORIZONS = [1, 3, 5]

# Seed pour reproductibilité
RANDOM_SEED = 42

# Colonnes clés
COL_PIPE_ID = "GID"  # ID du tronçon dans la table patrimoine
COL_PIPE_ID_ANOMALIE = "GID_OBJET"  # ID du tronçon dans la table anomalies
COL_DATE_INSTALL = "DDP"  # Date de pose
COL_DATE_HS = "DHS"  # Date hors service (abandon/défaillance)
COL_MATERIAU = "MAT"
COL_DIAMETRE = "DIAMETRE"
COL_LONGUEUR = "LNG"
COL_STATUT = "STATUT_OBJET"
COL_DATE_ANOMALIE = "DATE_DETECTION_parsed"
COL_TYPE_ANOMALIE = "TYPE_ANOMALIE"

# Seuils business pour évaluation
TOP_K_PERCENTAGES = [0.05, 0.10, 0.20]  # 5%, 10%, 20%
