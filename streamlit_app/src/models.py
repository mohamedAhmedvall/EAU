"""
Modèles de données pour Optiplan
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import json
import uuid


@dataclass
class Troncon:
    """Représente un tronçon de canalisation"""
    gid: str
    materiau: str
    diametre: float  # mm
    longueur: float  # mètres
    annee_pose: int
    statut: str
    age_jours: int
    score_ml: float = 0.0  # Score du modèle ML (0-1)
    score_opportunite: float = 0.0  # Score opportunité travaux (0-1)
    cout_unitaire: float = 150.0  # €/m par défaut
    classe_risque: int = 1  # 1-5

    @property
    def cout_total(self) -> float:
        """Coût total de renouvellement du tronçon"""
        return self.longueur * self.cout_unitaire

    @property
    def longueur_km(self) -> float:
        """Longueur en km"""
        return self.longueur / 1000

    @property
    def age_annees(self) -> float:
        """Âge en années"""
        return self.age_jours / 365.25


@dataclass
class TronconPlanifie:
    """Tronçon sélectionné pour renouvellement"""
    troncon: Troncon
    annee_renouvellement: int
    score_composite: float


@dataclass
class KpiAnnee:
    """KPIs pour une année donnée"""
    annee: int
    budget_consomme: float
    lineaire_km: float
    nb_troncons: int
    risque_moyen: float


@dataclass
class ResultatOptimisation:
    """Résultat d'une optimisation"""
    troncons_planifies: List[TronconPlanifie]
    budget_total_consomme: float
    lineaire_total_km: float
    risque_initial: float
    risque_residuel: float
    kpis_par_annee: List[KpiAnnee]
    message: str = ""
    succes: bool = True


@dataclass
class Scenario:
    """Scénario de simulation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    nom: str = ""
    description: str = ""
    horizon_ans: int = 1  # Modèle ML fiable à horizon 1 an
    budget_annuel: float = 1_500_000
    lineaire_min_pct: float = 1.0  # % du réseau
    lineaire_max_pct: float = 2.5  # % du réseau
    alpha: float = 1.0  # Poids risque ML vs opportunité
    statut: str = "brouillon"  # brouillon, termine
    date_creation: str = field(default_factory=lambda: datetime.now().isoformat())
    date_modification: str = field(default_factory=lambda: datetime.now().isoformat())
    resultat: Optional[ResultatOptimisation] = None

    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour sauvegarde JSON"""
        data = {
            "id": self.id,
            "nom": self.nom,
            "description": self.description,
            "horizon_ans": self.horizon_ans,
            "budget_annuel": self.budget_annuel,
            "lineaire_min_pct": self.lineaire_min_pct,
            "lineaire_max_pct": self.lineaire_max_pct,
            "alpha": self.alpha,
            "statut": self.statut,
            "date_creation": self.date_creation,
            "date_modification": self.date_modification,
        }
        if self.resultat:
            data["resultat"] = {
                "budget_total_consomme": self.resultat.budget_total_consomme,
                "lineaire_total_km": self.resultat.lineaire_total_km,
                "risque_initial": self.resultat.risque_initial,
                "risque_residuel": self.resultat.risque_residuel,
                "succes": self.resultat.succes,
                "message": self.resultat.message,
                "nb_troncons": len(self.resultat.troncons_planifies),
                "kpis_par_annee": [
                    {
                        "annee": k.annee,
                        "budget_consomme": k.budget_consomme,
                        "lineaire_km": k.lineaire_km,
                        "nb_troncons": k.nb_troncons,
                        "risque_moyen": k.risque_moyen
                    }
                    for k in self.resultat.kpis_par_annee
                ]
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Scenario":
        """Crée un scénario depuis un dictionnaire"""
        scenario = cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            nom=data.get("nom", ""),
            description=data.get("description", ""),
            horizon_ans=data.get("horizon_ans", 15),
            budget_annuel=data.get("budget_annuel", 1_500_000),
            lineaire_min_pct=data.get("lineaire_min_pct", 1.0),
            lineaire_max_pct=data.get("lineaire_max_pct", 2.5),
            alpha=data.get("alpha", 1.0),
            statut=data.get("statut", "brouillon"),
            date_creation=data.get("date_creation", datetime.now().isoformat()),
            date_modification=data.get("date_modification", datetime.now().isoformat()),
        )
        return scenario


@dataclass
class RegleCout:
    """Règle de coût par matériau/diamètre"""
    materiau: str
    diametre_min: float
    diametre_max: float
    cout_unitaire: float  # €/m


@dataclass
class ParametresCouts:
    """Paramètres de coûts de renouvellement"""
    cout_defaut: float = 150.0  # €/m
    regles: List[RegleCout] = field(default_factory=list)

    def get_cout(self, materiau: str, diametre: float) -> float:
        """Retourne le coût unitaire pour un matériau et diamètre donnés"""
        for regle in self.regles:
            if (regle.materiau == materiau and
                regle.diametre_min <= diametre <= regle.diametre_max):
                return regle.cout_unitaire
        return self.cout_defaut

    def to_dict(self) -> dict:
        return {
            "cout_defaut": self.cout_defaut,
            "regles": [
                {
                    "materiau": r.materiau,
                    "diametre_min": r.diametre_min,
                    "diametre_max": r.diametre_max,
                    "cout_unitaire": r.cout_unitaire
                }
                for r in self.regles
            ]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParametresCouts":
        return cls(
            cout_defaut=data.get("cout_defaut", 150.0),
            regles=[
                RegleCout(**r) for r in data.get("regles", [])
            ]
        )


@dataclass
class ParametresSeuils:
    """Seuils de classement des tronçons"""
    s3: float = 25.0  # Seuil entre C2 (Bon) et C3 (Moyen)
    s4: float = 50.0  # Seuil entre C3 et C4 (Mauvais)
    s5: float = 75.0  # Seuil entre C4 et C5 (Critique)

    def get_classe(self, score: float) -> int:
        """Retourne la classe (1-5) pour un score donné (0-100)"""
        score_100 = score * 100 if score <= 1 else score
        if score_100 == 0:
            return 1  # Non pertinent
        elif score_100 <= self.s3:
            return 2  # Bon état
        elif score_100 <= self.s4:
            return 3  # État moyen
        elif score_100 <= self.s5:
            return 4  # Mauvais état
        else:
            return 5  # Critique

    def get_classe_label(self, classe: int) -> str:
        """Retourne le label de la classe"""
        labels = {
            1: "Non Pertinent",
            2: "Bon état",
            3: "État moyen",
            4: "Mauvais état",
            5: "Critique"
        }
        return labels.get(classe, "Inconnu")

    def to_dict(self) -> dict:
        return {"s3": self.s3, "s4": self.s4, "s5": self.s5}

    @classmethod
    def from_dict(cls, data: dict) -> "ParametresSeuils":
        return cls(
            s3=data.get("s3", 25.0),
            s4=data.get("s4", 50.0),
            s5=data.get("s5", 75.0)
        )
