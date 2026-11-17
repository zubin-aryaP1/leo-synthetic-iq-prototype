"""Minimal swappable observation repository. SQLite is just one implementation."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol
import json
import sqlite3
import threading


class ObservationRepository(Protocol):
    def create_run(self, run_id: str, scenario: str, created_at: str) -> None: ...
    def save_batch(self, rows: list[dict]) -> None: ...
    def history(self, run_id: str, limit: int = 300) -> list[dict]: ...
    def counts(self) -> dict: ...


class SQLiteObservationRepository:
    def __init__(self, path="data/demo.sqlite3"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.conn:
            self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
               id TEXT PRIMARY KEY, scenario TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations (
               run_id TEXT NOT NULL REFERENCES runs(id),
               elapsed_s REAL NOT NULL, track_id INTEGER NOT NULL,
               frequency_hz REAL NOT NULL, frequency_rate_hz_s REAL NOT NULL,
               snr_db REAL NOT NULL, match_name TEXT, rmse_hz REAL,
               confidence TEXT, PRIMARY KEY(run_id, elapsed_s, track_id));
            CREATE INDEX IF NOT EXISTS ix_observations_run_time
                ON observations(run_id, elapsed_s);
            """)

    def create_run(self, run_id, scenario, created_at):
        with self.lock, self.conn:
            self.conn.execute("INSERT INTO runs VALUES(?,?,?)", (run_id, scenario, created_at))

    def save_batch(self, rows):
        if not rows:
            return
        values = [(
            row["run_id"], row["elapsed_s"], row["id"], row["frequency_hz"],
            row["frequency_rate_hz_s"], row["snr_db"],
            row["match"]["satellite"] if row["match"] else None,
            row["match"]["rmse_hz"] if row["match"] else None,
            row["match"]["confidence"] if row["match"] else None,
        ) for row in rows]
        with self.lock, self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?,?,?,?,?)", values)

    def history(self, run_id, limit=300):
        limit = max(1, min(int(limit), 2000))
        with self.lock:
            results = self.conn.execute(
                "SELECT * FROM (SELECT * FROM observations WHERE run_id=? "
                "ORDER BY elapsed_s DESC LIMIT ?) ORDER BY elapsed_s, track_id",
                (run_id, limit),
            ).fetchall()
        return [dict(row) for row in results]

    def counts(self):
        with self.lock:
            runs = self.conn.execute("SELECT count(*) FROM runs").fetchone()[0]
            observations = self.conn.execute("SELECT count(*) FROM observations").fetchone()[0]
        return {"runs": runs, "observations": observations}

    def close(self):
        with self.lock:
            self.conn.close()
