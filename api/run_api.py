"""
Script de lancement de l'API.
"""
import uvicorn
import sys
from pathlib import Path

# Ajouter le dossier parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

if __name__ == "__main__":
    print("=" * 60)
    print("DEMARRAGE DE L'API - Priorisation Canalisations")
    print("=" * 60)
    print("\nEndpoints disponibles:")
    print("  - Documentation Swagger: http://localhost:8000/docs")
    print("  - Documentation ReDoc:   http://localhost:8000/redoc")
    print("  - Health check:          http://localhost:8000/health")
    print("  - Prediction:            POST http://localhost:8000/predict")
    print("\nAppuyez sur Ctrl+C pour arreter le serveur")
    print("=" * 60 + "\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
