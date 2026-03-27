import os
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
ORIGINAL_DATASET_FILE = DATA_FOLDER / "original_dataset.csv"
MAIN_DATASET_FILE = DATA_FOLDER / "main_dataset.csv"
WEATHER_FILE = WEATHER_FOLDER / "weather.csv"

DAY_OFF_COLUMNS = [
    "is_fr_public_holiday",
    "is_fr_school_holiday_zone_a",
    "is_dest_public_holiday",
    "is_dest_school_holiday",
    "is_weekend",
]

WEATHER_FEATURE_COLUMNS = [
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "windspeed_10m_max",
]

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
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        match = lowered.get(candidate.lower())
        if match is not None:
            return match
    return None


def _load_airports_dataset() -> pd.DataFrame:
    airports_candidates = [
        "upply-airports.csv",
        "uply-airports.csv",
        "upply_airports.csv",
        "airports.csv",
    ]
    airports_path = _find_holiday_csv(HOLIDAYS_FOLDER, airports_candidates)
    if airports_path is None:
        print(
            f"Warning: no airport mapping CSV found in {HOLIDAYS_FOLDER}. "
            "Destination country holiday features will default to 0."
        )
        return pd.DataFrame(columns=["AirportPrevious", "dest_country"])

    airports_raw = pd.read_csv(airports_path, sep=None, engine="python")
    code_column = _pick_column(
        airports_raw.columns,
        ["code", "airportprevious", "airport_code", "airportcode", "iata", "iata_code"],
    )
    country_column = _pick_column(
        airports_raw.columns,
        ["country_code", "country_iso_code", "dest_country", "country"],
    )

    if code_column is None or country_column is None:
        print(
            f"Warning: airports CSV '{airports_path.name}' does not contain expected columns. "
            "Destination country holiday features will default to 0."
        )
        return pd.DataFrame(columns=["AirportPrevious", "dest_country"])

    airports = (
        airports_raw.rename(columns={code_column: "AirportPrevious", country_column: "dest_country"})[
            ["AirportPrevious", "dest_country"]
        ]
        .dropna(subset=["AirportPrevious"])
        .drop_duplicates()
    )
    print(f"Loaded airport mapping from {airports_path}")
    return airports


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
            "Destination country holiday features will default to 0."
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
            "Destination country holiday features will default to 0."
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
    print(f"Loaded destination holidays from {holidays_path}")
    return holidays


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
    main_dataset["LTScheduledDatetime"] = pd.to_datetime(main_dataset["LTScheduledDatetime"], errors="coerce")
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

    derived_columns = [
        "day_of_week",
        "is_weekend",
        "date",
        "date_only",
        "season",
        "dest_country",
        "is_fr_public_holiday",
        "is_fr_school_holiday_zone_a",
        "is_dest_public_holiday",
        "is_dest_school_holiday",
        "is_any_day_off",
        "scheduled_date",
        "days_until_next_day_off",
        "days_until_next_workday",
    ]
    flights = flights.drop(columns=[column for column in derived_columns if column in flights.columns])

    flights["LTScheduledDatetime"] = pd.to_datetime(flights["LTScheduledDatetime"], errors="coerce")
    flights["day_of_week"] = flights["LTScheduledDatetime"].dt.dayofweek
    flights["is_weekend"] = flights["day_of_week"].isin([5, 6]).astype(int)
    flights["date"] = flights["LTScheduledDatetime"].dt.normalize()
    flights["date_only"] = flights["LTScheduledDatetime"].dt.date
    flights["season"] = flights["LTScheduledDatetime"].dt.month.apply(_get_season)

    if not airports.empty and "AirportPrevious" in flights.columns:
        flights = flights.merge(airports, on="AirportPrevious", how="left")
    else:
        flights["dest_country"] = np.nan

    years = sorted(flights["LTScheduledDatetime"].dt.year.dropna().astype(int).unique())

    fr_public_holidays = set()
    for year in years:
        holidays_dict = JoursFeries.for_year(year, zone="Métropole")
        fr_public_holidays.update(holidays_dict.values())
    flights["is_fr_public_holiday"] = flights["date_only"].isin(fr_public_holidays).astype(int)

    school_holidays = SchoolHolidayDates()
    fr_school_holidays_zone_a = set()
    for year in years:
        year_holidays = school_holidays.holidays_for_year_and_zone(year, "A")
        fr_school_holidays_zone_a.update(year_holidays.keys())
    flights["is_fr_school_holiday_zone_a"] = flights["date_only"].isin(fr_school_holidays_zone_a).astype(int)

    if holidays.empty:
        dest_public = pd.DataFrame(columns=["dest_country", "date", "is_dest_public_holiday"])
        dest_school = pd.DataFrame(columns=["dest_country", "date", "is_dest_school_holiday"])
    else:
        holidays = holidays.copy()
        holidays["type"] = holidays["type"].replace({"public_holiday": "public", "school_holiday": "school"})
        holidays["start_date"] = pd.to_datetime(holidays["start_date"], errors="coerce")
        holidays["end_date"] = pd.to_datetime(holidays["end_date"], errors="coerce")
        holidays = holidays.dropna(subset=["country_code", "type", "start_date", "end_date"])
        holidays["date"] = holidays.apply(
            lambda row: pd.date_range(row["start_date"], row["end_date"], freq="D"),
            axis=1,
        )
        holiday_days = holidays.explode("date")[["country_code", "type", "date"]].drop_duplicates()

        dest_public = (
            holiday_days[holiday_days["type"] == "public"][["country_code", "date"]]
            .rename(columns={"country_code": "dest_country"})
            .drop_duplicates()
            .assign(is_dest_public_holiday=1)
        )
        dest_school = (
            holiday_days[holiday_days["type"] == "school"][["country_code", "date"]]
            .rename(columns={"country_code": "dest_country"})
            .drop_duplicates()
            .assign(is_dest_school_holiday=1)
        )

    flights = flights.merge(dest_public, on=["dest_country", "date"], how="left")
    flights = flights.merge(dest_school, on=["dest_country", "date"], how="left")

    flights["is_dest_public_holiday"] = flights["is_dest_public_holiday"].fillna(0).astype(int)
    flights["is_dest_school_holiday"] = flights["is_dest_school_holiday"].fillna(0).astype(int)
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
    apply_preprocessing()


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

    flights.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Preprocessed dataset saved to {output_path}")


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

    if "dest_country" not in processed.columns:
        processed["dest_country"] = np.nan

    processed["scheduled_date"] = processed["LTScheduledDatetime"].dt.normalize()
    processed["dest_country"] = processed["dest_country"].fillna("__UNKNOWN__")

    day_off_flags = processed[DAY_OFF_COLUMNS].fillna(0).astype(int)
    processed["is_any_day_off"] = day_off_flags.max(axis=1).astype(int)

    daily_calendar = (
        processed[["dest_country", "scheduled_date", "is_any_day_off"]]
        .dropna(subset=["scheduled_date"])
        .groupby(["dest_country", "scheduled_date"], as_index=False)["is_any_day_off"]
        .max()
        .sort_values(["dest_country", "scheduled_date"])
        .reset_index(drop=True)
    )

    countdown_frames = [
        compute_group_day_off_countdowns(group) for _, group in daily_calendar.groupby("dest_country", sort=False)
    ]

    if countdown_frames:
        countdowns = pd.concat(countdown_frames, ignore_index=True)
        processed = processed.merge(countdowns, on=["dest_country", "scheduled_date"], how="left")
    else:
        processed["days_until_next_day_off"] = np.nan
        processed["days_until_next_workday"] = np.nan

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
    return group[["dest_country", "scheduled_date", "days_until_next_day_off", "days_until_next_workday"]]


if __name__ == "__main__":
    load_dotenv()
    ensure_data_directories()
    load_main_dataset()
    merge_datasets()
