import os
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from google.cloud import bigquery
from google.cloud.bigquery.table import Row
from jours_feries_france import JoursFeries
from utils import ProgressBar
from vacances_scolaires_france import SchoolHolidayDates


DATA_FOLDER = Path("data")
WEATHER_FOLDER = DATA_FOLDER / "weather"
HOLIDAYS_FOLDER = DATA_FOLDER / "holidays"
WORLD_EVENTS_FOLDER = DATA_FOLDER / "world_events"
ORIGINAL_DATASET_FILE = DATA_FOLDER / "original_dataset.csv"
MAIN_DATASET_FILE = DATA_FOLDER / "main_dataset.csv"
WEATHER_FILE = WEATHER_FOLDER / "weather.csv"
WORLD_EVENTS_AIRPORTS_FILE = WORLD_EVENTS_FOLDER / "clean_airports.parquet"
WORLD_EVENTS_RISK_FILE = WORLD_EVENTS_FOLDER / "smoothed_risk_events.parquet"

DAY_OFF_COLUMNS = [
    "is_fr_public_holiday",
    "is_fr_school_holiday_zone_a",
    "is_origin_public_holiday",
    "is_origin_school_holiday",
    "is_dest_public_holiday",
    "is_dest_school_holiday",
    "is_weekend",
]

LYON_COUNTRY_CODE = "FR"

WEATHER_FEATURE_COLUMNS = [
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]

LAG_ROLLING_COLUMNS = [
    "pax_lag_1",
    "pax_lag_7",
    "pax_lag_30",
    "pax_rolling_7",
    "pax_ewm_7",
    "pax_diff_1",
    "pax_diff_7",
]

# Lyon Saint-Exupéry coordinates (fallback if not found in airports CSV)
LYON_AIRPORT_CODE = "LYS"
LYON_LAT_FALLBACK = 45.725556
LYON_LON_FALLBACK = 4.8414

WEATHER_BASE_PARAMS = {
    "latitude": 45.7589,
    "longitude": 4.8414,
    "daily": ",".join(WEATHER_FEATURE_COLUMNS),
    "timezone": "Europe/Paris",
}


def ensure_data_directories() -> None:
    DATA_FOLDER.mkdir(exist_ok=True)
    WEATHER_FOLDER.mkdir(exist_ok=True)
    HOLIDAYS_FOLDER.mkdir(exist_ok=True)
    WORLD_EVENTS_FOLDER.mkdir(exist_ok=True)


def _normalize_filename(filename: str) -> str:
    return "".join(character for character in filename.lower() if character.isalnum())


def _find_holiday_csv(folder: Path, candidate_names: list[str]) -> Path | None:
    for file_name in candidate_names:
        direct_path = folder / file_name
        if direct_path.exists():
            return direct_path

    candidate_tokens = {_normalize_filename(name) for name in candidate_names}
    for csv_path in folder.glob("*.csv"):
        if _normalize_filename(csv_path.name) in candidate_tokens:
            return csv_path

    return None


def _pick_column(columns: pd.Index, candidates: list[str]) -> str | None:
    lowered = {str(column).replace("\ufeff", "").strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _normalize_text_key(value: object) -> str | None:
    if pd.isna(value):
        return None

    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = normalized.encode("ascii", "ignore").decode("ascii").strip().lower()
    return normalized or None


def _normalize_airport_code_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})


def _normalize_country_code_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.upper()
    return normalized.replace({"": pd.NA, "NAN": pd.NA, "NONE": pd.NA, "<NA>": pd.NA})


def _add_world_event_risk_features(flights: pd.DataFrame) -> pd.DataFrame:
    enriched = flights.copy()

    stale_columns = [
        column_name
        for column_name in ["risk_score", "world_event_country_code", "world_event_date"]
        if column_name in enriched.columns
    ]
    if stale_columns:
        enriched = enriched.drop(columns=stale_columns)

    if "AirportPrevious" not in enriched.columns or "LTScheduledDatetime" not in enriched.columns:
        enriched["risk_score"] = 0.0
        return enriched

    if not WORLD_EVENTS_AIRPORTS_FILE.exists() or not WORLD_EVENTS_RISK_FILE.exists():
        print(
            "Warning: world event parquet files not found in " f"{WORLD_EVENTS_FOLDER}. Risk score will default to 0.0."
        )
        enriched["risk_score"] = 0.0
        return enriched

    airports = pd.read_parquet(WORLD_EVENTS_AIRPORTS_FILE, columns=["iata_code", "iso_country"])
    risk_events = pd.read_parquet(WORLD_EVENTS_RISK_FILE, columns=["iso_country", "Date", "SmoothedRisk"])

    airports["iata_code"] = _normalize_airport_code_series(airports["iata_code"])
    airports["iso_country"] = _normalize_country_code_series(airports["iso_country"])

    risk_events["iso_country"] = _normalize_country_code_series(risk_events["iso_country"])
    risk_events["Date"] = pd.to_datetime(risk_events["Date"], errors="coerce").dt.normalize()
    risk_events["SmoothedRisk"] = pd.to_numeric(risk_events["SmoothedRisk"], errors="coerce")
    risk_events = (
        risk_events.rename(columns={"iso_country": "world_event_country_code", "Date": "world_event_date"})
        .dropna(subset=["world_event_country_code", "world_event_date"])
        .drop_duplicates(subset=["world_event_country_code", "world_event_date"], keep="last")
    )

    enriched["AirportPrevious"] = _normalize_airport_code_series(enriched["AirportPrevious"])
    enriched["world_event_date"] = pd.to_datetime(enriched["LTScheduledDatetime"], errors="coerce").dt.normalize()

    airport_lookup = airports.rename(
        columns={
            "iata_code": "AirportPrevious",
            "iso_country": "world_event_country_code",
        }
    ).drop_duplicates(subset=["AirportPrevious"], keep="first")

    enriched = enriched.merge(airport_lookup, on="AirportPrevious", how="left")
    enriched = enriched.merge(
        risk_events,
        on=["world_event_country_code", "world_event_date"],
        how="left",
    )

    enriched["risk_score"] = pd.to_numeric(enriched["SmoothedRisk"], errors="coerce").fillna(0.0)
    enriched = enriched.drop(columns=["SmoothedRisk", "world_event_country_code", "world_event_date"], errors="ignore")
    return enriched


def _normalize_direction_value(value: object) -> str | None:
    normalized = _normalize_text_key(value)
    letters_only = "".join(character for character in normalized or "" if character.isalpha())
    if letters_only.startswith("arriv") or normalized in {"arrivee", "arrival"}:
        return "arrivee"
    if letters_only.startswith(("depart", "dpart")) or normalized in {"depart", "departure"}:
        return "depart"
    return normalized


def _load_airports_dataset() -> pd.DataFrame:
    if not WORLD_EVENTS_AIRPORTS_FILE.exists():
        print(
            f"Warning: airport mapping parquet not found at {WORLD_EVENTS_AIRPORTS_FILE}. "
            "Country-dependent route features will default to missing values."
        )
        return pd.DataFrame(columns=["airport_code", "country_code"])

    airports = (
        pd.read_parquet(WORLD_EVENTS_AIRPORTS_FILE, columns=["iata_code", "iso_country"])
        .rename(columns={"iata_code": "airport_code", "iso_country": "country_code"})[["airport_code", "country_code"]]
        .assign(
            airport_code=lambda df: _normalize_airport_code_series(df["airport_code"]),
            country_code=lambda df: _normalize_country_code_series(df["country_code"]),
        )
        .dropna(subset=["airport_code", "country_code"])
        .drop_duplicates()
    )
    print(f"Loaded airport mapping from {WORLD_EVENTS_AIRPORTS_FILE}")
    return airports


def _load_airports_with_coords() -> pd.DataFrame:
    """Load airports CSV keeping latitude and longitude columns."""
    airports_candidates = [
        "upply-airports.csv",
        "airports.csv",
    ]
    airports_path = _find_holiday_csv(HOLIDAYS_FOLDER, airports_candidates)
    if airports_path is None:
        return pd.DataFrame(columns=["airport_code", "latitude", "longitude"])

    airports_raw = pd.read_csv(airports_path, sep=None, engine="python")
    code_column = _pick_column(
        airports_raw.columns,
        ["code", "airportprevious", "airport_code", "airportcode", "iata", "iata_code"],
    )
    lat_column = _pick_column(airports_raw.columns, ["latitude", "lat"])
    lon_column = _pick_column(airports_raw.columns, ["longitude", "lon", "lng"])

    if code_column is None or lat_column is None or lon_column is None:
        return pd.DataFrame(columns=["airport_code", "latitude", "longitude"])

    airports = (
        airports_raw.rename(
            columns={
                code_column: "airport_code",
                lat_column: "latitude",
                lon_column: "longitude",
            }
        )[["airport_code", "latitude", "longitude"]]
        .assign(
            airport_code=lambda df: _normalize_airport_code_series(df["airport_code"]),
            latitude=lambda df: pd.to_numeric(df["latitude"], errors="coerce"),
            longitude=lambda df: pd.to_numeric(df["longitude"], errors="coerce"),
        )
        .dropna(subset=["airport_code", "latitude", "longitude"])
        .drop_duplicates(subset=["airport_code"])
    )
    return airports


def _haversine_km(
    lat_series: pd.Series,
    lon_series: pd.Series,
    fixed_lat: float,
    fixed_lon: float,
) -> pd.Series:
    """Vectorised Haversine distance (km) from each row to a fixed point."""
    R = 6371.0
    lat1_r = np.radians(lat_series)
    lat2_r = np.radians(fixed_lat)
    dlat = lat2_r - lat1_r
    dlon = np.radians(fixed_lon - lon_series)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return pd.Series(2 * R * np.arcsin(np.sqrt(a.clip(0, 1))), index=lat_series.index)


def add_distance_features(flights: pd.DataFrame) -> pd.DataFrame:
    """
    Add flight_distance_km: great-circle distance (km) between Lyon (LYS) and
    the remote airport (AirportPrevious).

    Continuous distance is preferred over a binary is_long_haul flag because
    tree-based models learn their own split thresholds; the discrete version
    loses information without any benefit for those algorithms.
    """
    processed = flights.copy()

    if "flight_distance_km" in processed.columns:
        processed = processed.drop(columns=["flight_distance_km"])

    if "AirportPrevious" not in processed.columns:
        processed["flight_distance_km"] = np.nan
        return processed

    airports_coords = _load_airports_with_coords()
    if airports_coords.empty:
        processed["flight_distance_km"] = np.nan
        return processed

    # Get Lyon's coordinates from the CSV; fall back to hardcoded values.
    lyon_row = airports_coords.loc[airports_coords["airport_code"] == LYON_AIRPORT_CODE]
    lyon_lat = float(lyon_row["latitude"].iloc[0]) if not lyon_row.empty else LYON_LAT_FALLBACK
    lyon_lon = float(lyon_row["longitude"].iloc[0]) if not lyon_row.empty else LYON_LON_FALLBACK

    remote_coords = airports_coords.rename(
        columns={
            "airport_code": "AirportPrevious",
            "latitude": "_remote_lat",
            "longitude": "_remote_lon",
        }
    )
    processed = processed.merge(remote_coords, on="AirportPrevious", how="left")

    has_coords = processed["_remote_lat"].notna() & processed["_remote_lon"].notna()
    distance = pd.Series(np.nan, index=processed.index)
    if has_coords.any():
        distance.loc[has_coords] = _haversine_km(
            processed.loc[has_coords, "_remote_lat"],
            processed.loc[has_coords, "_remote_lon"],
            lyon_lat,
            lyon_lon,
        )
    processed["flight_distance_km"] = distance
    processed = processed.drop(columns=["_remote_lat", "_remote_lon"], errors="ignore")
    return processed


def _load_destination_holidays_dataset() -> pd.DataFrame:
    holidays_candidates = [
        "openholidays_2023_2027.csv",
        "openholidays.csv",
        "open_holidays_2023_2027.csv",
    ]
    holidays_path = _find_holiday_csv(HOLIDAYS_FOLDER, holidays_candidates)
    if holidays_path is None:
        print(
            f"Warning: no destination holiday CSV found in {HOLIDAYS_FOLDER}. "
            "Origin/destination holiday features will default to 0."
        )
        return pd.DataFrame(columns=["country_code", "type", "start_date", "end_date"])

    holidays_raw = pd.read_csv(holidays_path, sep=None, engine="python")
    country_column = _pick_column(holidays_raw.columns, ["country_iso_code", "country_code", "dest_country"])
    type_column = _pick_column(holidays_raw.columns, ["holiday_type", "type"])
    start_column = _pick_column(holidays_raw.columns, ["start_date", "startdate", "date_start"])
    end_column = _pick_column(holidays_raw.columns, ["end_date", "enddate", "date_end"])

    if None in [country_column, type_column, start_column, end_column]:
        print(
            f"Warning: destination holidays CSV '{holidays_path.name}' does not contain expected columns. "
            "Origin/destination holiday features will default to 0."
        )
        return pd.DataFrame(columns=["country_code", "type", "start_date", "end_date"])

    holidays = holidays_raw.rename(
        columns={
            country_column: "country_code",
            type_column: "type",
            start_column: "start_date",
            end_column: "end_date",
        }
    )[["country_code", "type", "start_date", "end_date"]]
    print(f"Loaded holiday calendar from {holidays_path}")
    return holidays


def _add_route_country_features(flights: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    enriched = flights.copy()

    if "AirportPrevious" not in enriched.columns:
        enriched["AirportPrevious"] = pd.NA
    if "AirportOrigin" not in enriched.columns:
        enriched["AirportOrigin"] = pd.NA

    enriched["AirportPrevious"] = _normalize_airport_code_series(enriched["AirportPrevious"])
    enriched["AirportOrigin"] = _normalize_airport_code_series(enriched["AirportOrigin"])

    previous_lookup = airports.rename(columns={"airport_code": "AirportPrevious", "country_code": "remote_country"})
    origin_lookup = airports.rename(columns={"airport_code": "AirportOrigin", "country_code": "leg_country"})

    enriched = enriched.merge(previous_lookup, on="AirportPrevious", how="left")
    enriched = enriched.merge(origin_lookup, on="AirportOrigin", how="left")

    if "Direction" in enriched.columns:
        direction = enriched["Direction"].map(_normalize_direction_value)
    else:
        direction = pd.Series(pd.NA, index=enriched.index, dtype="object")

    is_arrival = direction.eq("arrivee").fillna(False)
    is_departure = direction.eq("depart").fillna(False)

    enriched["origin_country"] = pd.Series(pd.NA, index=enriched.index, dtype="object")
    enriched.loc[is_arrival, "origin_country"] = enriched.loc[is_arrival, "remote_country"]
    enriched.loc[is_departure, "origin_country"] = LYON_COUNTRY_CODE

    enriched["dest_country"] = pd.Series(pd.NA, index=enriched.index, dtype="object")
    enriched.loc[is_arrival, "dest_country"] = LYON_COUNTRY_CODE
    enriched.loc[is_departure, "dest_country"] = enriched.loc[is_departure, "remote_country"]

    is_domestic = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
    known_leg_country = enriched["leg_country"].notna()
    is_domestic.loc[known_leg_country] = (
        enriched.loc[known_leg_country, "leg_country"].eq(LYON_COUNTRY_CODE).astype("Int64")
    )
    enriched["is_domestic"] = is_domestic

    is_route_domestic = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
    known_remote_country = enriched["remote_country"].notna()
    is_route_domestic.loc[known_remote_country] = (
        enriched.loc[known_remote_country, "remote_country"].eq(LYON_COUNTRY_CODE).astype("Int64")
    )
    enriched["is_route_domestic"] = is_route_domestic

    has_stopover = pd.Series(pd.NA, index=enriched.index, dtype="Int64")
    stopover_mask = enriched["AirportPrevious"].notna() & enriched["AirportOrigin"].notna()
    has_stopover.loc[stopover_mask] = (
        enriched.loc[stopover_mask, "AirportPrevious"] != enriched.loc[stopover_mask, "AirportOrigin"]
    ).astype("Int64")
    enriched["has_stopover"] = has_stopover

    return enriched


def _add_country_holiday_features(
    flights: pd.DataFrame,
    public_holiday_days: pd.DataFrame,
    school_holiday_days: pd.DataFrame,
    *,
    country_column: str,
    public_column: str,
    school_column: str,
) -> pd.DataFrame:
    enriched = flights.merge(
        public_holiday_days.rename(columns={"country_code": country_column}).assign(**{public_column: 1}),
        on=[country_column, "date"],
        how="left",
    )
    enriched = enriched.merge(
        school_holiday_days.rename(columns={"country_code": country_column}).assign(**{school_column: 1}),
        on=[country_column, "date"],
        how="left",
    )

    france_mask = enriched[country_column].eq(LYON_COUNTRY_CODE).fillna(False)
    enriched.loc[france_mask, public_column] = enriched.loc[france_mask, "is_fr_public_holiday"]
    enriched.loc[france_mask, school_column] = enriched.loc[france_mask, "is_fr_school_holiday_zone_a"]

    enriched[public_column] = enriched[public_column].fillna(0).astype(int)
    enriched[school_column] = enriched[school_column].fillna(0).astype(int)
    return enriched


def _build_weather_datasets() -> pd.DataFrame:
    archive_end_date = pd.Timestamp.now(tz="Europe/Paris").date() - timedelta(days=2)

    archive_response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            **WEATHER_BASE_PARAMS,
            "start_date": "2023-01-01",
            "end_date": archive_end_date.isoformat(),
        },
        timeout=60,
    )
    archive_response.raise_for_status()
    archive_json = archive_response.json()

    forecast_response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            **WEATHER_BASE_PARAMS,
            "past_days": 30,
            "forecast_days": 16,
        },
        timeout=60,
    )
    forecast_response.raise_for_status()
    forecast_json = forecast_response.json()

    archive_daily = archive_json.get("daily")
    forecast_daily = forecast_json.get("daily")
    if archive_daily is None or forecast_daily is None:
        raise RuntimeError("Open-Meteo response is missing the 'daily' payload.")

    df_archive = pd.DataFrame(archive_daily)
    df_forecast = pd.DataFrame(forecast_daily)

    weather_dataset = pd.concat([df_archive, df_forecast], ignore_index=True)
    weather_dataset = weather_dataset.drop_duplicates(subset=["time"], keep="first")
    weather_dataset = weather_dataset.sort_values("time").reset_index(drop=True)
    return weather_dataset


def _drop_existing_weather_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    weather_columns_to_drop = [
        column_name
        for column_name in dataset.columns
        if column_name in WEATHER_FEATURE_COLUMNS
        or (column_name.endswith("_x") and column_name[:-2] in WEATHER_FEATURE_COLUMNS)
        or (column_name.endswith("_y") and column_name[:-2] in WEATHER_FEATURE_COLUMNS)
    ]
    if weather_columns_to_drop:
        return dataset.drop(columns=weather_columns_to_drop)
    return dataset


def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_main_dataset() -> None:
    """
    Download the main dataset from BigQuery if it is not already present locally.
    """

    ensure_data_directories()

    if MAIN_DATASET_FILE.exists():
        print(f"\tMain dataset already exists at {MAIN_DATASET_FILE}. Skipping download.")
        return

    project_id = _get_required_env("PROJECT_ID")
    dataset_id = _get_required_env("DATASET_ID")
    table_id = _get_required_env("TABLE_ID")
    service_account_key_path = os.getenv("SERVICE_ACCOUNT_KEY_PATH") or os.getenv("YOUR_SERVICE_ACCOUNT_KEY_PATH")

    if service_account_key_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

    table_ref = f"`{project_id}.{dataset_id}.{table_id}`"
    query = f"SELECT * FROM {table_ref};"

    client = bigquery.Client(project=project_id)
    print(f"\tRunning BigQuery query: {query}\n")

    query_job = client.query(query)
    rows: Iterator[Row] = query_job.result()
    total_rows = getattr(rows, "total_rows", 0) or 0

    print(f"\tQuery completed. {total_rows} rows retrieved.")

    progress_bar = None
    if total_rows > 0:
        progress_bar = ProgressBar(total_rows, update_every=max(1, total_rows // 100))
        progress_bar.start()

    dataset_rows = []
    for row in rows:
        dataset_rows.append(dict(row.items()))
        if progress_bar is not None:
            progress_bar.increment()

    if progress_bar is not None:
        progress_bar.finish()

    dataset = pd.DataFrame(dataset_rows)
    dataset.to_csv(MAIN_DATASET_FILE, index=False, encoding="utf-8-sig")
    dataset.to_csv(ORIGINAL_DATASET_FILE, index=False, encoding="utf-8-sig")

    print(f"\tMain dataset saved to {MAIN_DATASET_FILE}")


def load_weather_data() -> None:
    ensure_data_directories()

    if WEATHER_FILE.exists():
        print(f"Weather dataset already exists at {WEATHER_FILE}. Skipping download.")
        weather_dataset = pd.read_csv(WEATHER_FILE)
    else:
        print("Fetching weather archive and forecast data...")
        weather_dataset = _build_weather_datasets()
        weather_dataset.to_csv(WEATHER_FILE, index=False, encoding="utf-8-sig")
        print(f"Weather dataset saved to {WEATHER_FILE}")

    if not MAIN_DATASET_FILE.exists():
        raise FileNotFoundError(f"Main dataset not found at {MAIN_DATASET_FILE}")

    main_dataset = pd.read_csv(MAIN_DATASET_FILE)

    # 2. Convert the main column (in dataframe) to datetime so we can manipulate it
    main_dataset["LTScheduledDatetime"] = pd.to_datetime(
        main_dataset["LTScheduledDatetime"], format="%Y-%m-%d %H:%M:%S"
    )

    # Drop any existing weather columns to avoid duplicate suffix conflicts on re-run
    weather_cols = ["precipitation_sum", "rain_sum", "snowfall_sum", "windspeed_10m_max"]
    existing_weather_cols = [
        c for c in main_dataset.columns if any(c == w or c.startswith(w + "_") for w in weather_cols)
    ]
    main_dataset = main_dataset.drop(columns=existing_weather_cols)

    # 3. Extract JUST the date into a temporary string column (e.g., '2026-04-22')
    main_dataset["Temp_Date_Match"] = main_dataset["LTScheduledDatetime"].dt.strftime("%Y-%m-%d")

    weather_dataset = weather_dataset.copy()
    weather_dataset["time"] = pd.to_datetime(weather_dataset["time"], errors="coerce").dt.strftime("%Y-%m-%d")

    main_dataset = _drop_existing_weather_columns(main_dataset)
    df_merged = pd.merge(main_dataset, weather_dataset, left_on="Temp_Date_Match", right_on="time", how="left")
    df_merged = df_merged.drop(columns=["Temp_Date_Match", "time"], errors="ignore")
    df_merged.to_csv(MAIN_DATASET_FILE, index=False, encoding="utf-8-sig")

    print(f"Success! Saved {len(weather_dataset)} days of continuous weather data.")


def load_holiday_data() -> None:
    if not MAIN_DATASET_FILE.exists():
        raise FileNotFoundError(f"Main dataset not found at {MAIN_DATASET_FILE}")

    flights = pd.read_csv(MAIN_DATASET_FILE)
    airports = _load_airports_dataset()
    holidays = _load_destination_holidays_dataset()

    stale_columns = [
        "day_of_week",
        "is_weekend",
        "date",
        "date_only",
        "season",
        "remote_country",
        "leg_country",
        "origin_country",
        "dest_country",
        "is_domestic",
        "is_route_domestic",
        "has_stopover",
        "is_fr_public_holiday",
        "is_bridge_day",
        "is_fr_school_holiday_zone_a",
        "is_fr_school_holiday_zone_b",
        "is_fr_school_holiday_zone_c",
        "is_first_last_day_of_school_holiday",
        "is_origin_public_holiday",
        "is_origin_school_holiday",
        "is_dest_public_holiday",
        "is_dest_school_holiday",
        "is_any_day_off",
        "scheduled_date",
        "days_until_next_day_off",
        "days_until_next_workday",
    ]
    flights = flights.drop(columns=[column for column in stale_columns if column in flights.columns])

    flights["LTScheduledDatetime"] = pd.to_datetime(flights["LTScheduledDatetime"], errors="coerce")
    flights["day_of_week"] = flights["LTScheduledDatetime"].dt.dayofweek
    flights["is_weekend"] = flights["day_of_week"].isin([5, 6]).astype(int)
    flights["date"] = flights["LTScheduledDatetime"].dt.normalize()
    flights["date_only"] = flights["LTScheduledDatetime"].dt.date

    flights["season"] = flights["LTScheduledDatetime"].dt.month.apply(_get_season)

    # =========================
    # 3. MAPPING AIRPORT -> COUNTRY
    # =========================

    flights = _add_route_country_features(flights, airports)

    # =========================
    # 4. FRANCE : PUBLIC HOLIDAY
    # =========================

    years = sorted(flights["LTScheduledDatetime"].dt.year.dropna().astype(int).unique())

    fr_public_holidays = set()
    for year in years:
        holidays_dict = JoursFeries.for_year(year, zone="Métropole")
        fr_public_holidays.update(holidays_dict.values())
    flights["is_fr_public_holiday"] = flights["date_only"].isin(fr_public_holidays).astype(int)

    # is_bridge_day: a weekday sandwiched between a public holiday and a weekend
    # Friday (dow=4) whose Thursday was a public holiday, or Monday (dow=0) whose Tuesday will be a public holiday
    fr_holiday_ts = set(pd.to_datetime(list(fr_public_holidays)))
    flight_dates_ts = pd.to_datetime(flights["date_only"])
    prev_day_is_holiday = (flight_dates_ts - pd.Timedelta(days=1)).isin(fr_holiday_ts)
    next_day_is_holiday = (flight_dates_ts + pd.Timedelta(days=1)).isin(fr_holiday_ts)
    flights["is_bridge_day"] = (
        ((flights["day_of_week"] == 4) & prev_day_is_holiday) | ((flights["day_of_week"] == 0) & next_day_is_holiday)
    ).astype(int)

    # =========================
    # 5. FRANCE : SCHOOL HOLIDAY
    # =========================
    # Lyon = zone A

    school_holidays = SchoolHolidayDates()
    fr_school_holidays_zone_a = set()
    for year in years:
        year_holidays = school_holidays.holidays_for_year_and_zone(year, "A")
        fr_school_holidays_zone_a.update(year_holidays.keys())
    flights["is_fr_school_holiday_zone_a"] = flights["date_only"].isin(fr_school_holidays_zone_a).astype(int)

    fr_school_holidays_zone_b = set()
    for y in years:
        year_holidays = school_holidays.holidays_for_year_and_zone(y, "B")
        fr_school_holidays_zone_b.update(year_holidays.keys())
    flights["is_fr_school_holiday_zone_b"] = flights["date_only"].isin(fr_school_holidays_zone_b).astype(int)

    fr_school_holidays_zone_c = set()
    for y in years:
        year_holidays = school_holidays.holidays_for_year_and_zone(y, "C")
        fr_school_holidays_zone_c.update(year_holidays.keys())
    flights["is_fr_school_holiday_zone_c"] = flights["date_only"].isin(fr_school_holidays_zone_c).astype(int)

    # Mark the first or last day of a contiguous school-holiday block across zones B and C.
    all_bc_holidays_ts = set(pd.to_datetime(list(fr_school_holidays_zone_b | fr_school_holidays_zone_c)))
    first_last_bc_ts = {
        d
        for d in all_bc_holidays_ts
        if (d - pd.Timedelta(days=1)) not in all_bc_holidays_ts or (d + pd.Timedelta(days=1)) not in all_bc_holidays_ts
    }
    flights["is_first_last_day_of_school_holiday"] = (
        pd.to_datetime(flights["date_only"]).isin(first_last_bc_ts)
    ).astype(int)

    # =========================
    # 6. ORIGIN / DESTINATION : HOLIDAYS
    # =========================

    holidays = holidays.rename(columns={"country_iso_code": "country_code", "holiday_type": "type"})

    holidays["country_code"] = _normalize_country_code_series(holidays["country_code"])
    holidays["type"] = holidays["type"].astype("string").str.strip().str.lower()
    holidays["type"] = holidays["type"].replace({"public_holiday": "public", "school_holiday": "school"})

    holidays["start_date"] = pd.to_datetime(holidays["start_date"], errors="coerce")
    holidays["end_date"] = pd.to_datetime(holidays["end_date"], errors="coerce")

    holidays = holidays[["country_code", "type", "start_date", "end_date"]].dropna(
        subset=["country_code", "type", "start_date", "end_date"]
    )

    holidays["date"] = [
        list(pd.date_range(start, end, freq="D")) for start, end in zip(holidays["start_date"], holidays["end_date"])
    ]

    holiday_days = holidays.explode("date")[["country_code", "type", "date"]].drop_duplicates()

    public_holiday_days = holiday_days[holiday_days["type"] == "public"][["country_code", "date"]].drop_duplicates()
    school_holiday_days = holiday_days[holiday_days["type"] == "school"][["country_code", "date"]].drop_duplicates()

    flights = _add_country_holiday_features(
        flights,
        public_holiday_days,
        school_holiday_days,
        country_column="origin_country",
        public_column="is_origin_public_holiday",
        school_column="is_origin_school_holiday",
    )
    flights = _add_country_holiday_features(
        flights,
        public_holiday_days,
        school_holiday_days,
        country_column="dest_country",
        public_column="is_dest_public_holiday",
        school_column="is_dest_school_holiday",
    )

    flights = flights.drop(columns=["date", "date_only"], errors="ignore")

    flights.to_csv(MAIN_DATASET_FILE, index=False, encoding="utf-8-sig")
    print(f"Holiday features saved to {MAIN_DATASET_FILE}")


def _get_season(month: float) -> str | None:
    if pd.isna(month):
        return None
    month = int(month)
    if month in [12, 1, 2]:
        return "winter"
    if month in [3, 4, 5]:
        return "spring"
    if month in [6, 7, 8]:
        return "summer"
    return "autumn"


def merge_datasets() -> None:
    load_weather_data()
    load_holiday_data()
    load_world_event_data()
    apply_preprocessing()


def load_world_event_data() -> None:
    if not MAIN_DATASET_FILE.exists():
        raise FileNotFoundError(f"Main dataset not found at {MAIN_DATASET_FILE}")

    flights = pd.read_csv(MAIN_DATASET_FILE)
    flights = _add_world_event_risk_features(flights)
    flights.to_csv(MAIN_DATASET_FILE, index=False, encoding="utf-8-sig")
    print(f"World event risk features saved to {MAIN_DATASET_FILE}")


def apply_preprocessing(
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    input_path = Path(input_path or MAIN_DATASET_FILE)
    output_path = Path(output_path or MAIN_DATASET_FILE)

    flights = pd.read_csv(input_path)
    flights["LTScheduledDatetime"] = pd.to_datetime(flights["LTScheduledDatetime"], errors="coerce")

    flights = add_cyclical_datetime_features(flights)
    flights = add_day_off_countdown_features(flights)
    flights = add_lag_and_rolling_features(flights)
    flights = add_distance_features(flights)

    flights.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Preprocessed dataset saved to {output_path}")


def add_lag_and_rolling_features(flights: pd.DataFrame) -> pd.DataFrame:
    processed = flights.copy()

    # drop stale columns from a previous run so we always recompute fresh.
    stale = [col for col in LAG_ROLLING_COLUMNS if col in processed.columns]
    if stale:
        processed = processed.drop(columns=stale)

    group_key = "FlightNumberNormalized"
    required = {"NbPaxTotal", "LTScheduledDatetime", group_key}
    if not required.issubset(processed.columns):
        for col in LAG_ROLLING_COLUMNS:
            processed[col] = np.nan
        return processed

    # ── Daily mean pax per (flight, date) ────────────────────────────────────
    processed["_temp_date"] = pd.to_datetime(processed["LTScheduledDatetime"]).dt.normalize()

    daily = (
        processed.dropna(subset=[group_key])
        .groupby([group_key, "_temp_date"])["NbPaxTotal"]
        .mean()
        .reset_index()
        .rename(columns={"NbPaxTotal": "_pax"})
        .sort_values([group_key, "_temp_date"])
        .reset_index(drop=True)
    )

    for n_days, col_name in [(1, "pax_lag_1"), (7, "pax_lag_7"), (30, "pax_lag_30")]:
        lag_lookup = daily[[group_key, "_temp_date", "_pax"]].copy()
        lag_lookup["_temp_date"] = lag_lookup["_temp_date"] + pd.Timedelta(days=n_days)
        lag_lookup = lag_lookup.rename(columns={"_pax": col_name})
        processed = processed.merge(
            lag_lookup,
            on=[group_key, "_temp_date"],
            how="left",
        )

    daily["_rolling_7"] = daily.groupby(group_key)["_pax"].transform(
        lambda s: s.shift(1).rolling(window=7, min_periods=1).mean()
    )

    # alpha = 2 / (span + 1) = 0.25 for span=7
    daily["_ewm_7"] = daily.groupby(group_key)["_pax"].transform(
        lambda s: s.shift(1).ewm(span=7, adjust=False, min_periods=1).mean()
    )
    daily["_diff_1"] = daily.groupby(group_key)["_pax"].transform(lambda s: s.shift(1).diff(1))
    daily["_diff_7"] = daily.groupby(group_key)["_pax"].transform(lambda s: s.shift(1).diff(7))

    rolling_lookup = daily[[group_key, "_temp_date", "_rolling_7", "_ewm_7", "_diff_1", "_diff_7"]].rename(
        columns={
            "_rolling_7": "pax_rolling_7",
            "_ewm_7": "pax_ewm_7",
            "_diff_1": "pax_diff_1",
            "_diff_7": "pax_diff_7",
        }
    )
    processed = processed.merge(rolling_lookup, on=[group_key, "_temp_date"], how="left")

    processed = processed.drop(columns=["_temp_date"], errors="ignore")
    return processed


def add_cyclical_datetime_features(flights: pd.DataFrame) -> pd.DataFrame:
    processed = flights.copy()
    scheduled = processed["LTScheduledDatetime"]

    processed["day_of_week"] = scheduled.dt.dayofweek
    processed["is_weekend"] = processed["day_of_week"].isin([5, 6]).astype(int)
    processed["month_of_year"] = scheduled.dt.month
    processed["hour_of_day"] = scheduled.dt.hour + scheduled.dt.minute / 60.0 + scheduled.dt.second / 3600.0

    processed["hour_sin"] = np.sin(2 * np.pi * processed["hour_of_day"] / 24.0)
    processed["hour_cos"] = np.cos(2 * np.pi * processed["hour_of_day"] / 24.0)
    processed["day_of_week_sin"] = np.sin(2 * np.pi * processed["day_of_week"] / 7.0)
    processed["day_of_week_cos"] = np.cos(2 * np.pi * processed["day_of_week"] / 7.0)
    processed["month_sin"] = np.sin(2 * np.pi * (processed["month_of_year"] - 1) / 12.0)
    processed["month_cos"] = np.cos(2 * np.pi * (processed["month_of_year"] - 1) / 12.0)

    return processed


def add_day_off_countdown_features(flights: pd.DataFrame) -> pd.DataFrame:
    processed = flights.copy()
    ensure_day_off_columns_exist(processed)

    stale_countdown_columns = [
        "scheduled_date",
        "is_any_day_off",
        "days_until_next_day_off",
        "days_until_next_workday",
    ]
    processed = processed.drop(columns=[column for column in stale_countdown_columns if column in processed.columns])

    if "origin_country" not in processed.columns:
        processed["origin_country"] = np.nan
    if "dest_country" not in processed.columns:
        processed["dest_country"] = np.nan

    processed["scheduled_date"] = processed["LTScheduledDatetime"].dt.normalize()
    processed["origin_country"] = processed["origin_country"].fillna("__UNKNOWN__")
    processed["dest_country"] = processed["dest_country"].fillna("__UNKNOWN__")

    day_off_flags = processed[DAY_OFF_COLUMNS].fillna(0).astype(int)
    processed["is_any_day_off"] = day_off_flags.max(axis=1).astype(int)

    daily_calendar = (
        processed[["origin_country", "dest_country", "scheduled_date", "is_any_day_off"]]
        .dropna(subset=["scheduled_date"])
        .groupby(["origin_country", "dest_country", "scheduled_date"], as_index=False)["is_any_day_off"]
        .max()
        .sort_values(["origin_country", "dest_country", "scheduled_date"])
        .reset_index(drop=True)
    )

    countdown_frames = [
        compute_group_day_off_countdowns(group)
        for _, group in daily_calendar.groupby(["origin_country", "dest_country"], sort=False)
    ]

    if countdown_frames:
        countdowns = pd.concat(countdown_frames, ignore_index=True)
        processed = processed.merge(countdowns, on=["origin_country", "dest_country", "scheduled_date"], how="left")
    else:
        processed["days_until_next_day_off"] = np.nan
        processed["days_until_next_workday"] = np.nan

    processed["origin_country"] = processed["origin_country"].replace("__UNKNOWN__", np.nan)
    processed["dest_country"] = processed["dest_country"].replace("__UNKNOWN__", np.nan)
    return processed


def ensure_day_off_columns_exist(flights: pd.DataFrame) -> None:
    missing_columns = [column_name for column_name in DAY_OFF_COLUMNS if column_name not in flights.columns]
    if missing_columns:
        raise RuntimeError(f"Missing day-off columns for preprocessing: {missing_columns}")


def compute_group_day_off_countdowns(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("scheduled_date").reset_index(drop=True).copy()
    dates = pd.to_datetime(group["scheduled_date"]).dt.date.tolist()
    is_day_off = group["is_any_day_off"].astype(bool).tolist()

    next_day_off_date = None
    next_workday_date = None
    days_until_next_day_off = [np.nan] * len(group)
    days_until_next_workday = [np.nan] * len(group)

    for index in range(len(group) - 1, -1, -1):
        current_date = dates[index]

        if is_day_off[index]:
            days_until_next_day_off[index] = 0
            if next_workday_date is not None:
                days_until_next_workday[index] = (next_workday_date - current_date).days
            next_day_off_date = current_date
        else:
            days_until_next_workday[index] = 0
            if next_day_off_date is not None:
                days_until_next_day_off[index] = (next_day_off_date - current_date).days
            next_workday_date = current_date

    group["days_until_next_day_off"] = days_until_next_day_off
    group["days_until_next_workday"] = days_until_next_workday
    return group[
        ["origin_country", "dest_country", "scheduled_date", "days_until_next_day_off", "days_until_next_workday"]
    ]


# --- Configuration et exécution du script ---
if __name__ == "__main__":
    load_dotenv()
    ensure_data_directories()

    load_main_dataset()
    merge_datasets()
