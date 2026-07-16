"""Data ingestion: download the SQLite database if needed and load its tables.

The pipeline reads from the relative path `data/delivery.db`; if the file is absent it is
downloaded from the configured blob URL, then every table is read into a DataFrame.
"""
import os
import sqlite3
import requests
import pandas as pd


def download_db(url, db_path):
    """Download the SQLite database to db_path if it is not already present.

    Parameters:
        url: source URL of the .db file.
        db_path: local destination path (its parent dir is created).
    Returns:
        str: the local db_path (existing or freshly downloaded).
    """
    if os.path.exists(db_path):
        return db_path
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(db_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return db_path


def load_tables(db_path, url=None, logger=None):
    """Load all user tables from the SQLite database into pandas DataFrames.

    Parameters:
        db_path: path to the SQLite file (downloaded from `url` if missing).
        url: optional source URL used when the file is not on disk.
        logger: optional logger for progress messages.
    Returns:
        dict: {table_name: DataFrame} for every non-internal table.
    """
    if not os.path.exists(db_path) and url:
        if logger: logger.info(f"Database not found locally; downloading from {url}")
        download_db(url, db_path)
    with sqlite3.connect(db_path) as conn:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        tables = {n: pd.read_sql(f"SELECT * FROM {n}", conn) for n in names}
    if logger: logger.info(f"Loaded tables {list(tables)} from {db_path}")
    return tables
