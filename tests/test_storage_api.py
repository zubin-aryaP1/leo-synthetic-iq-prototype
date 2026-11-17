from fastapi.testclient import TestClient

from app.storage import SQLiteObservationRepository
from app.main import app


def test_sqlite_repository(tmp_path):
    db = SQLiteObservationRepository(tmp_path / "test.sqlite3")
    db.create_run("run-1", "clean", "2026-11-09T00:00:00Z")
    db.save_batch([{
        "run_id": "run-1", "elapsed_s": 1.0, "id": 1,
        "frequency_hz": 123.0, "frequency_rate_hz_s": 2.5,
        "snr_db": 18.0, "match": None,
    }])
    assert db.counts()["observations"] == 1
    assert db.history("run-1")[0]["track_id"] == 1
    db.close()


def test_api_controls_and_history(tmp_path, monkeypatch):
    monkeypatch.setenv("LEO_DEMO_DB", str(tmp_path / "api.sqlite3"))
    with TestClient(app) as client:
        r = client.get("/api/status")
        assert r.status_code == 200
        assert r.json()["simulated"] is True
        assert client.post("/api/control", json={"action":"reset","scenario":"noisy"}).status_code == 200
        assert client.post("/api/control", json={"action":"unknown"}).status_code == 422
        assert client.post("/api/control", json={"action":"start"}).json()["running"]
        assert client.post("/api/control", json={"action":"stop"}).status_code == 200
        assert "observations" in client.get("/api/history").json()
        assert client.get("/").status_code == 200
