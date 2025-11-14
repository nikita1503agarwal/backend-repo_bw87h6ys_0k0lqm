"""
Database Helper Functions

MongoDB helper functions ready to use in your backend code.
Import and use these functions in your API endpoints for database operations.
Falls back to in-memory mock when DATABASE_URL/DATABASE_NAME are not set, so the
API remains reachable in sandboxes.
"""

import os
from datetime import datetime, timezone
from typing import Union
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

db = None

DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

try:
    if DATABASE_URL and DATABASE_NAME:
        from pymongo import MongoClient
        client = MongoClient(DATABASE_URL)
        db = client[DATABASE_NAME]
    else:
        # Fallback mock DB to keep API up in envs without Mongo
        import mongomock
        client = mongomock.MongoClient()
        db = client["explorer_mock"]
except Exception:
    try:
        import mongomock
        client = mongomock.MongoClient()
        db = client["explorer_mock"]
    except Exception:
        db = None


def create_document(collection_name: str, data: Union[BaseModel, dict]):
    if db is None:
        raise Exception("Database not available")
    if isinstance(data, BaseModel):
        data = data.model_dump()
    data = {**data}
    now = datetime.now(timezone.utc)
    data.setdefault("created_at", now)
    data.setdefault("updated_at", now)
    result = db[collection_name].insert_one(data)
    return str(result.inserted_id)


def get_documents(collection_name: str, filter_dict: dict | None = None, limit: int | None = None):
    if db is None:
        raise Exception("Database not available")
    cursor = db[collection_name].find(filter_dict or {})
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)
