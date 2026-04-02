import sqlite3
import unicodedata
from pathlib import Path
from sqlite3 import Connection, Cursor

import numpy as np
import pandas as pd


class SqliteManager:
    def __init__(self, db_path: str | Path, commit_on_exit: bool = True) -> None:
        self.db_path = db_path
        self.commit_on_exit = commit_on_exit

    def __enter__(self) -> tuple[Cursor, Connection]:
        self.connection = sqlite3.connect(self.db_path)
        self.db = self.connection.cursor()
        return self.db, self.connection

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.db.close()
        if self.commit_on_exit:
            self.connection.commit()
        self.connection.close()


def normalize_filename(filename: str) -> str:
    return "".join(character for character in filename.lower() if character.isalnum())


def find_csv_file(folder: Path, candidate_names: list[str]) -> Path | None:
    for file_name in candidate_names:
        candidate_path = folder / file_name
        if candidate_path.exists():
            return candidate_path

    candidate_tokens = {normalize_filename(name) for name in candidate_names}
    for csv_path in folder.glob("*.csv"):
        if normalize_filename(csv_path.name) in candidate_tokens:
            return csv_path

    return None


def pick_column(columns: pd.Index, candidates: list[str]) -> str | None:
    normalized = {str(column).replace("\ufeff", "").strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match is not None:
            return match
    return None


def normalize_text_key(value: object) -> str | None:
    if pd.isna(value):
        return None

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").strip().lower()
    return ascii_text or None


def normalize_airport_code_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})


def normalize_country_code_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})


def normalize_direction_value(value: object) -> str | None:
    normalized = normalize_text_key(value)
    letters_only = "".join(character for character in normalized or "" if character.isalpha())
    if letters_only.startswith("arriv") or normalized in {"arrivee", "arrival"}:
        return "arrivee"
    if letters_only.startswith(("depart", "dpart")) or normalized in {"depart", "departure"}:
        return "depart"
    return normalized


def get_season(month: float) -> str | None:
    if pd.isna(month):
        return None

    month = int(month)
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def haversine_km(
    lat_series: pd.Series,
    lon_series: pd.Series,
    fixed_latitude: float,
    fixed_longitude: float,
) -> pd.Series:
    earth_radius_km = 6371.0
    lat1_radians = np.radians(lat_series)
    lat2_radians = np.radians(fixed_latitude)
    delta_latitude = lat2_radians - lat1_radians
    delta_longitude = np.radians(fixed_longitude - lon_series)
    haversine_term = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(lat1_radians) * np.cos(lat2_radians) * np.sin(delta_longitude / 2) ** 2
    )
    return pd.Series(2 * earth_radius_km * np.arcsin(np.sqrt(haversine_term.clip(0, 1))), index=lat_series.index)
