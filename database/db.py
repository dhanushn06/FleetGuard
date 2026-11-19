
import sqlite3
from pathlib import Path
from datetime import datetime

# Find the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Database file location
DB_PATH = BASE_DIR / "fleetguard.db"


# =====================================
# 1. CREATE DATABASE AND TABLE
# =====================================

def initialize_db():

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS checkins (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            timestamp TEXT NOT NULL,

            driver_id TEXT NOT NULL,

            bpm REAL,

            quality_score REAL,

            status TEXT NOT NULL,

            reasons TEXT

        )
    """)

    connection.commit()
    connection.close()


# =====================================
# 2. SAVE DRIVER CHECK-IN
# =====================================

def save_checkin(record):

    initialize_db()

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO checkins
        (timestamp, driver_id, bpm, quality_score, status, reasons)

        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        record.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        record["driver_id"],
        record.get("bpm"),
        record.get("quality_score"),
        record["status"],
        record.get("reasons", "")
    ))

    connection.commit()
    connection.close()


# =====================================
# 3. GET ALL DRIVER RECORDS
# =====================================

def get_all_checkins():

    initialize_db()

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT * FROM checkins
        ORDER BY id DESC
    """)

    records = [dict(row) for row in cursor.fetchall()]

    connection.close()

    return records


# =====================================
# 4. GET DASHBOARD COUNTS
# =====================================

def get_summary():

    records = get_all_checkins()

    total = len(records)

    accepted = sum(
        1 for record in records
        if record["status"] == "Accepted"
    )

    rejected = sum(
        1 for record in records
        if record["status"] == "Rejected"
    )

    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected
    }


# Create database automatically when this file runs
if __name__ == "__main__":

    initialize_db()

    print("PulseCam FleetGuard database created successfully!")
    print("Database location:", DB_PATH)
