"""
Script de test de l'API.
Peut être utilisé pour tester l'API localement.
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test du health check."""
    print("\n=== Test Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_model_info():
    """Test des infos du modèle."""
    print("\n=== Test Model Info ===")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict_single():
    """Test de prédiction pour un seul tronçon."""
    print("\n=== Test Prediction Single ===")

    # Tronçon de test
    pipe = {
        "GID": 410000670,
        "DDP": "1963-01-01",
        "DHS": None,
        "MAT": "FT",
        "DIAMETRE": 300,
        "LNG": 71.43
    }

    response = requests.post(
        f"{BASE_URL}/predict/single",
        params={"freeze_date": "2024-01-01"},
        json=pipe
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_predict_batch():
    """Test de prédiction batch."""
    print("\n=== Test Prediction Batch ===")

    request_data = {
        "pipes": [
            {"GID": 410966779, "DDP": "2009-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 100, "LNG": 82.7},
            {"GID": 410000670, "DDP": "1963-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 300, "LNG": 71.4},
            {"GID": 410001234, "DDP": "1949-01-01", "DHS": None, "MAT": "FTG", "DIAMETRE": 100, "LNG": 14.9},
            {"GID": 410001246, "DDP": "1979-01-01", "DHS": None, "MAT": "BTM", "DIAMETRE": 1000, "LNG": 46.6}
        ],
        "anomalies": [
            {"GID_OBJET": 410000670, "DATE_DETECTION": "2015-03-15", "TYPE_ANOMALIE": "FUITE_SIGNAL_TR"},
            {"GID_OBJET": 410000670, "DATE_DETECTION": "2018-07-22", "TYPE_ANOMALIE": "FUITE_DETECT_TR"},
            {"GID_OBJET": 410001234, "DATE_DETECTION": "2010-01-10", "TYPE_ANOMALIE": "FUITE_SIGNAL_TR"}
        ],
        "freeze_date": "2024-01-01",
        "horizon_years": 1
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=request_data
    )
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"Total pipes: {result.get('total_pipes')}")
    print(f"Summary: {json.dumps(result.get('summary', {}), indent=2)}")
    print("\nPredictions:")
    for pred in result.get('predictions', []):
        print(f"  GID {pred['GID']}: score={pred['risk_score']:.3f}, rank={pred['rank']}, category={pred['risk_category']}")

    return response.status_code == 200


def test_predict_top_k():
    """Test du top-k."""
    print("\n=== Test Top-K Prediction ===")

    request_data = {
        "pipes": [
            {"GID": 410966779, "DDP": "2009-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 100, "LNG": 82.7},
            {"GID": 410000670, "DDP": "1963-01-01", "DHS": None, "MAT": "FT", "DIAMETRE": 300, "LNG": 71.4},
            {"GID": 410001234, "DDP": "1949-01-01", "DHS": None, "MAT": "FTG", "DIAMETRE": 100, "LNG": 14.9},
            {"GID": 410001246, "DDP": "1979-01-01", "DHS": None, "MAT": "BTM", "DIAMETRE": 1000, "LNG": 46.6}
        ],
        "anomalies": [],
        "freeze_date": "2024-01-01",
        "horizon_years": 1,
        "top_k_percent": 0.50  # Top 50% pour ce petit exemple
    }

    response = requests.post(
        f"{BASE_URL}/predict/top-k",
        json=request_data
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def main():
    """Exécute tous les tests."""
    print("=" * 60)
    print("TESTS DE L'API")
    print("=" * 60)
    print(f"URL de base: {BASE_URL}")

    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Predict Single", test_predict_single),
        ("Predict Batch", test_predict_batch),
        ("Predict Top-K", test_predict_top_k),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except requests.exceptions.ConnectionError:
            print(f"\nERREUR: Impossible de se connecter a {BASE_URL}")
            print("Assurez-vous que l'API est demarree avec: python run_api.py")
            return
        except Exception as e:
            print(f"\nERREUR: {e}")
            results.append((name, "ERROR"))

    print("\n" + "=" * 60)
    print("RESULTATS DES TESTS")
    print("=" * 60)
    for name, status in results:
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
