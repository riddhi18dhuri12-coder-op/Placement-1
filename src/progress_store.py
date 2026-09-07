"""
progress_store.py
------------------
Lightweight local persistence so a student can save each assessment run
and see their placement-probability / skill scores trend over time.

Storage: a single SQLite file at data/progress.db. Each row is keyed by a
student-chosen "Student ID" (e.g. roll number or name) — this is a simple
way to separate one student's history from another's, NOT a real
authentication system. There is no password or identity verification, so
don't rely on this for anything sensitive; it's meant for a single
classroom/lab deployment where students self-report a consistent ID.
"""

import os
import sqlite3
import datetime
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "progress.db")


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            branch TEXT,
            probability REAL,
            inputs_json TEXT
        )
    """)
    return conn


def save_snapshot(student_id: str, branch: str, probability: float, inputs: dict):
    """Persist one assessment run for a student."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO progress (student_id, timestamp, branch, probability, inputs_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            student_id.strip(),
            datetime.datetime.now().isoformat(timespec="seconds"),
            branch,
            float(probability),
            json.dumps(inputs),
        ),
    )
    conn.commit()
    conn.close()


def load_history(student_id: str) -> list:
    """Return all saved snapshots for a student, oldest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT timestamp, branch, probability, inputs_json FROM progress "
        "WHERE student_id = ? ORDER BY timestamp ASC",
        (student_id.strip(),),
    ).fetchall()
    conn.close()
    history = []
    for ts, branch, prob, inputs_json in rows:
        entry = {"timestamp": ts, "branch": branch, "probability": prob}
        entry.update(json.loads(inputs_json))
        history.append(entry)
    return history


def clear_history(student_id: str):
    conn = _get_conn()
    conn.execute("DELETE FROM progress WHERE student_id = ?", (student_id.strip(),))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    save_snapshot("TEST123", "CSE", 0.55, {"CGPA": 7.0, "Technical_Skills": 50})
    save_snapshot("TEST123", "CSE", 0.63, {"CGPA": 7.2, "Technical_Skills": 60})
    for h in load_history("TEST123"):
        print(h)
    clear_history("TEST123")
