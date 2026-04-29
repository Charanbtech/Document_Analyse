"""
Database Integration Module
Supports MongoDB Atlas (primary) and SQLite (fallback) for prediction storage.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "database" / "predictions.db"


class DatabaseHandler:
    """
    Handles prediction storage with MongoDB or SQLite fallback.
    """

    def __init__(self, mongo_uri: str = None):
        self.mongo_client = None
        self.mongo_db = None
        self.use_mongo = False

        # Try MongoDB first
        if mongo_uri:
            try:
                import pymongo
                self.mongo_client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
                self.mongo_client.server_info()  # Test connection
                self.mongo_db = self.mongo_client["dc_ml_db"]
                self.use_mongo = True
                logger.info("Connected to MongoDB Atlas")
            except Exception as e:
                logger.warning(f"MongoDB connection failed, using SQLite: {e}")

        # Always init SQLite as fallback
        self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite database and create table if not exists."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence REAL,
                source TEXT DEFAULT 'text',
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info(f"SQLite database ready at {DB_PATH}")

    def store_prediction(self, input_text: str, predicted_category: str,
                         confidence: float = None, source: str = "text") -> bool:
        """
        Store a prediction record.

        Args:
            input_text: The original input text (truncated for storage)
            predicted_category: The predicted document class
            confidence: Prediction confidence score (0-1)
            source: Input source ('text' or 'file')

        Returns:
            True if stored successfully
        """
        record = {
            "input_text": input_text[:500],  # Limit text length
            "predicted_category": predicted_category,
            "confidence": round(confidence, 4) if confidence else None,
            "source": source,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Try MongoDB
        if self.use_mongo:
            try:
                self.mongo_db["predictions"].insert_one(record)
                logger.info("Prediction stored in MongoDB")
                return True
            except Exception as e:
                logger.warning(f"MongoDB write failed, falling back to SQLite: {e}")

        # SQLite fallback
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.execute(
                "INSERT INTO predictions (input_text, predicted_category, confidence, source, timestamp) VALUES (?,?,?,?,?)",
                (record["input_text"], record["predicted_category"],
                 record["confidence"], record["source"], record["timestamp"])
            )
            conn.commit()
            conn.close()
            logger.info("Prediction stored in SQLite")
            return True
        except Exception as e:
            logger.error(f"SQLite write failed: {e}")
            return False

    def get_recent_predictions(self, limit: int = 20) -> list:
        """Retrieve recent predictions from the database."""
        if self.use_mongo:
            try:
                cursor = self.mongo_db["predictions"].find(
                    {}, {"_id": 0}
                ).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.warning(f"MongoDB read failed: {e}")

        # SQLite fallback
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.error(f"SQLite read failed: {e}")
            return []

    def get_stats(self) -> dict:
        """Get prediction statistics."""
        if self.use_mongo:
            try:
                total = self.mongo_db["predictions"].count_documents({})
                pipeline = [{"$group": {"_id": "$predicted_category", "count": {"$sum": 1}}}]
                by_cat = {d["_id"]: d["count"] for d in self.mongo_db["predictions"].aggregate(pipeline)}
                return {"total": total, "by_category": by_cat, "db": "MongoDB"}
            except Exception:
                pass

        try:
            conn = sqlite3.connect(str(DB_PATH))
            total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
            rows = conn.execute(
                "SELECT predicted_category, COUNT(*) as cnt FROM predictions GROUP BY predicted_category"
            ).fetchall()
            conn.close()
            return {"total": total, "by_category": {r[0]: r[1] for r in rows}, "db": "SQLite"}
        except Exception as e:
            return {"total": 0, "by_category": {}, "db": "SQLite", "error": str(e)}
